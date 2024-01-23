# BandWidth_Test
Test the GPU bandwidth of  collectives operators like all-reduce, all-gather, broadcast and all-to-all primitives on single-node multi-GPUs (2, 4, 8 cards) and multi-nodes multi-GPUs (16 cards) setups, using only PyTorch and Python built-in packages.

The results:

single-node multi-GPUs：

![img_v3_027b_9455c6ff-991a-4310-8f2e-e07bf84a217g](https://github.com/wangbluo/BandWidth_Test/assets/32676639/7daba81b-8a9e-4c13-82f0-30b778653025)
![img_v3_027b_50e90d59-f163-4beb-bfda-eb5e9557e4bg](https://github.com/wangbluo/BandWidth_Test/assets/32676639/c1b5ab2a-c0be-4f9b-bebf-def51cd23272)
![img_v3_027b_236ccabe-af10-4343-96f0-feeb153c08ag](https://github.com/wangbluo/BandWidth_Test/assets/32676639/4f8e9d4b-1d71-4466-aba3-f64afa4fdb1b)

multi-nodes multi-GPUs：
Run run_test_bandwidth.sh script. 
For example, run command """sh run_test_bandwidth.sh 2 2 0 10.20.1.81 22 """ on the first node, and run command """sh run_test_bandwidth.sh 2 2 1 10.20.1.81 22 """ on the second node. The $NNODES represents the machine you want to use and $NODE_RANK represents the gpus you want to use per node machine.  
![img_v3_027b_44e5f429-6e6f-4912-8151-aaa6825030cg](https://github.com/wangbluo/BandWidth_Test/assets/32676639/e29a14f0-0234-4d71-b685-76502c942731)
![img_v3_027b_484d8993-21bc-4671-b624-2a8b6e1cf58g](https://github.com/wangbluo/BandWidth_Test/assets/32676639/d9ad0939-768b-455f-bccf-92f30897bc29)


