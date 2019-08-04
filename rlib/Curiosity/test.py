from Curiosity_LSTM import main

if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'VentureDeterministic-v4', 'SpaceInvadersDeterministic-v4',]
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1']
    #env_id_list = ['SuperMarioBros-1-1-v0']
    for env_id in env_id_list:
        main(env_id)