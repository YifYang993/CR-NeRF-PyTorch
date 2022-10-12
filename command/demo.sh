#!/bin/bash
 
# Declare an array of string with type
declare -a StringArray=("Linux Mint" "Fedora" "Red Hat Linux" "Ubuntu" "Debian" )
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   echo $val
done