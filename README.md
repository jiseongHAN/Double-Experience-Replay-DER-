# Double-Experience-Replay-DER-

<b>Pytorch implementaion of Double_Experience_Replay (DER)</b>

This method mixes two stratesgies for sampling experiences which will be stored in replay buffer.\
You could choose strategies whatever you want, but this paper we use temperal difference (TD) value based sample strategy and uniform sample strategy.



# Contents
This implementaion contains:

<b>Simulation of Urban MObility (SUMO) </b>
* Lane change Environmnet
* Ring Network Environment

*YeongDong Bridge Environment does not supported.

![example](asset/4.png)
![example2](asset/5.png)

# Method

Using Uniform sample strategy and TD value based sampling method. \
As a training algorithm we use Deep Q-learning (DQN)
