import pykitti



def get_dataset(num_images: int):

    basedir = 'data'
    sequence = '00'
    # data_dir = './data/sequences/00'  # Try KITTI_sequence_2 t oo
    frames = range(0, num_images, 1) #Indicate how many frames to use
    dataset = pykitti.odometry(basedir, sequence, frames=frames)#, frames=frames)

    return dataset