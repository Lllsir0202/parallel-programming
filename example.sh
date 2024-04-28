# example.sh
#!/bin/sh
# PBS -N simd

pssh -h $PBS_NODEFILE mkdir -p /home/s2212878/SIMD 1>&2
scp master:/home/s2212878/SIMD/simd /home/s2212878/SIMD
pscp -h $PBS_NODEFILE master:/home/s2212878/SIMD/simd /home/s2212878/SIMD 1>&2
/home/s2212878/SIMD/simd
