import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays_cross_ray']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples



def render_rays_cross_ray(models,
                embeddings,
                rays,
                ts,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, models, xyz, z_vals, test_time=False,fine=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        args=kwargs['args']
        if fine:
            model=models['fine']
        else:model=models['coarse']
        typ = model.typ
        N_samples_ = xyz.shape[1] ##torch.Size([8192, 64, 3])
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)
        if args.pertubeCord==True:
            pertube_ratio=0.00001
            xyz_+=pertube_ratio*torch.rand(xyz_.size(),device=xyz_.device)
        # Perform model inference to get cross-ray features
        B = xyz_.shape[0]
        out_chunks = []
        dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
    
        for i in range(0, B, chunk):
            inputs = [embedding_xyz(xyz_[i:i+chunk]), dir_embedded_[i:i+chunk]]
            feature_of_nerf=model(torch.cat(inputs, 1), output_random=output_random)
            #get cross-ray features
            out_chunks += [feature_of_nerf]

        out = torch.cat(out_chunks, 0) 
        out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
        static_rgbs = out[..., :args.nerf_out_dim] # (N_rays, N_samples_, 3)
        static_sigmas = out[...,args.nerf_out_dim] # (N_rays, N_samples_)
        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        noise = torch.randn_like(static_sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(static_sigmas+noise))

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        weights = alphas * transmittance

        results[f'weights_{typ}'] = weights

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                          'n1 n2 c -> n1 c', 'sum')
        results[f'feature_{typ}'] = rgb_map
        
        if output_random:
            results[f'feature_fine_random'] = rgb_map

        results[f'depth_{typ}'] = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        return


    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
    results = {}
    output_random = False
    inference(results, models, xyz_coarse, z_vals, test_time, **kwargs)
    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        model = models['fine']

        output_random = kwargs.get('output_random', True) and model.encode_random

        inference(results, models, xyz_fine, z_vals, test_time,fine=True, **kwargs)

    return results


def docode(results, models,type,**kwargs):
        feature_coarse=results['rgb_'+type] #torch.Size([699008, 4])
        lastdim=feature_coarse.size(-1)
        feature_coarse = rearrange(feature_coarse, 'n1 n3 -> n3 n1', n3=lastdim)
        feature_coarse = rearrange(feature_coarse, ' n3 (h w) ->  1 n3 h w',  h=int(kwargs['H']), w=int(kwargs['W']),n3=lastdim)  ##torch.Size([1, 64, 340, 514])
        rgbs_pred_coarse=models['decoder'](feature_coarse, kwargs['a_embedded_from_img'])
        rgbs_pred_coarse=rearrange(rgbs_pred_coarse, ' 1 n1 h w ->  (h w) n1',  h=int(kwargs['H']), w=int(kwargs['W']), n1=3)
        results['rgb_'+type]=rgbs_pred_coarse
        return results