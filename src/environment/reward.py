class Reward:
    # OPTIMISED
    INIT = -1
    SUBNET = 2
    DEVICE = 2
    COMPROMISED = 2
    SENSITIVE = 100
    HONEYPOT = -100
    
class RewardOptimised(Reward):
    INIT = -1
    SUBNET = 2
    DEVICE = 2
    COMPROMISED = 2
    SENSITIVE = 100
    HONEYPOT = -100

class RewardRisky(Reward):
    INIT = -1
    SUBNET = 2
    DEVICE = 2
    COMPROMISED = 2
    SENSITIVE = 100
    HONEYPOT = -5

class RewardLowRisk(Reward):
    INIT = -1
    SUBNET = 2
    DEVICE = 2
    COMPROMISED = 2
    SENSITIVE = 100
    HONEYPOT = -1000

class RewardExploration(Reward):
    INIT = -1
    SUBNET = 5
    DEVICE = 5
    COMPROMISED = 5
    SENSITIVE = 10
    HONEYPOT = -10