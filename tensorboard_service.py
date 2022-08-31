from tensorboard import program

# make tensorboard by given name(log path) and port
# append tensorboard url to reverse proxy
# return url
# set expire an hour
# kill in timer and delete in revers proxy
tb = program.TensorBoard()