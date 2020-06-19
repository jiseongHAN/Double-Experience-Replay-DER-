# Double Experience Replay (DER)

<b>Pytorch implementaion of Double_Experience_Replay (DER)</b>

This method mixes two stratesgies for sampling experiences which will be stored in replay buffer.\
You could choose strategies whatever you want, but this paper we use temperal difference (TD) value based sample strategy and uniform sample strategy.



# Contents
This implementaion contains:

<b>Simulation of Urban MObility (SUMO) </b>
* Lane change Environmnet
* Ring Network Environment

**YeongDong Bridge Environment does not supported.

<p float="left">
  <img src="asset/4.png" width="250px" height="200px"/>
  <img src="asset/5.png" width="250px" height="200px"/> 
  <img src="asset/ringex.png" width="250px" height="200px"/> 
</p>

# Method

Using Uniform sample strategy and TD value based sampling method. \
As a training algorithm we use Deep Q-learning (DQN)

# Requirements
* [Pytorch](https://pytorch.org)
* [Flow](https://flow-project.github.io/)
* Numpy
* [Gym](http://gym.openai.com/)
* TensorboardX


# Usage

To train SUMO with ring environment
```
cd ring
python ring.py
```

To train SUMO with Lane Change environment
```
cd lanechange
python lane.py
```

# Result

YeongDong Bridge Agent (LEFT, white car) Lane Change Agent (RIGHT, white car)
<p float="left">
  <img src="asset/yd.gif" width="400px" height="400px"/>
  <img src="asset/lane.gif" width="400px" height="400px"/> 
</p>

YeongDong Bridge (DQN, DER, PER)     Ring Network (DQN, DER, PER)

<p float="left">
  <img src="asset/yddg.png" width="400px" height="300px"/ title="Yeongdong Bridge">
  <img src="asset/ring.png" width="400px" height="300px"/ title="Ring Network"> 
</p>

