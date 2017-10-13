#!/bin/bash

declare -a kk_lis=(60 65)
declare -a ll_lis=(80)
declare -a jj_lis=(120)

for kk in "${kk_lis[@]}"; do
    for ll in "${ll_lis[@]}"; do
        for jj in "${jj_lis[@]}"; do
            qsub sh_coup${jj}_dkm${ll}_prec${kk}.sh
done
done
done
