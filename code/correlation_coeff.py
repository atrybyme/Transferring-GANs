import sys
import locale
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import time
from common.ops import *
from common.score import *
from common.data_loader import *
from common.logger import Logger

############################################################################################################################################

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist, toy")
tf.app.flags.DEFINE_string("sResultTag", "class_v0",
                           "your tag for each test case")

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR +
                           "result/", "where to save the checkpoint and sample")

tf.app.flags.DEFINE_boolean("bAMGAN", True, "")

tf.app.flags.DEFINE_integer("iMaxIter", 50000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 100, "")

tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")

tf.app.flags.DEFINE_float("fLrIni", 0.0004, "")
tf.app.flags.DEFINE_float("fBeta1", 0.5, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")

tf.app.flags.DEFINE_string("oDecay", 'linear', "exp, linear")
tf.app.flags.DEFINE_string("oOpt", 'adam', "adam, sgd, mom")
tf.app.flags.DEFINE_string("oAct", 'lrelu', "relu, lrelu, selu")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iDimsZ", 100, "")

tf.app.flags.DEFINE_integer("iFilterDimsG", 96, "")
tf.app.flags.DEFINE_integer("iFilterDimsD", 32, "")

tf.app.flags.DEFINE_float("fDropRate", 0.0, "")

cfg(sys.argv)

allocate_gpu()

############################################################################################################################################
real_datas = tf.placeholder(
    tf.float32, [None, cfg.iDimsC, 32, 32], name='real_datas')
real_labels = tf.placeholder(tf.int32, shape=[None])

num_logits = 10


iFilterDimsD = cfg.iFilterDimsD

with tf.variable_scope('discriminator', tf.AUTO_REUSE):

    h0 = real_datas
    h1 = noise(h0, 0.1, bAdd=False,bMulti = False)

    h2 = conv2d(h0, iFilterDimsD * 1, ksize=3,
                stride=1, name='conv32')  # 32x32
    h3 = tf.nn.leaky_relu(h2, name='h3')
    h4 = dropout(h3, cfg.fDropRate)

    h5 = conv2d(h4, iFilterDimsD * 2, ksize=3, stride=2,
                name='conv32_16')  # 32x32 --> 16x16
    h6 = batch_norm(h5, name='bn16')
    h7 = tf.nn.leaky_relu(h6, name='h7')
    h8 = dropout(h7, cfg.fDropRate)

    h9 = conv2d(h8, iFilterDimsD * 4, ksize=3, stride=2,
                name='conv16_8')  # 16x16 --> 8x8
    h10 = batch_norm(h9, name='bn8')
    h11 = tf.nn.leaky_relu(h10, name='h11')
    h12 = dropout(h11, cfg.fDropRate)

    h13 = conv2d(h12, iFilterDimsD * 8, ksize=3,
                 stride=2, name='conv8_4')  # 8x8 --> 4x4
    h14 = batch_norm(h13, name='bn4')
    h15 = tf.nn.leaky_relu(h14, name='h15')
    h16 = dropout(h15, cfg.fDropRate)

    h17 = avgpool(h16, h16.get_shape().as_list()[
        2], h16.get_shape().as_list()[3])
    h18 = tf.contrib.layers.flatten(h17)
    h19 = dropout(h18, cfg.fDropRate)

    h20 = linear(h19, num_logits)


############################################################################################################################################


def load_dataset(dataset_name):
    return load_cifar10() if dataset_name is "cifar10" else load_inception()


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        print(shape)
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count

############################################################################################################################################


dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
data_gen = data_gen_random(dataX, 260)

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag)
             else "") + cfg.sDataSet

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/samples/'
sCheckpointDir = sTestCaseDir + '/checkpoint/'

makedirs(cfg.sResultDir)
makedirs(sTestCaseDir)
makedirs(sSampleDir)
makedirs(sCheckpointDir)
makedirs(sTestCaseDir + '/code/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.log(sTestCaseDir)

commandline = ''
for arg in ['CUDA_VISIBLE_DEVICES="0" python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))

copydir(SOURCE_DIR + "code/", sTestCaseDir + '/source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + '/source/common/')

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

############################################################################################################################################


real_logits = h20

dis_total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=real_logits, labels=real_labels))

prediction = tf.cast(tf.argmax(real_logits, 1), tf.int32)
equality = tf.equal(prediction, real_labels)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]
for i in tot_vars:
    print(i.name)
global_step = tf.Variable(0, trainable=False, name='global_step')

lr = cfg.fLrIni * \
    tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))

##update learning rate
lr=0

dis_optimizer = tf.train.AdamOptimizer(
    learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
dis_gradient_values = dis_optimizer.compute_gradients(
    dis_total_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(
    dis_gradient_values, global_step=global_step)

saver = tf.train.Saver(max_to_keep=1000)

############################################################################################################################################

iter = 0
last_save_time = last_log_time = last_plot_time = last_score_time = time.time()


#######################


##Computation starts here.....
##Load Dataset....
checkpoint_dir = SOURCE_DIR + 'result/class_v5_cifar10/checkpoint'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
check_pnts = ckpt.all_model_checkpoint_paths
checking_data = data_gen.__next__()

##Start taking output
#RAM Problem so have to take new output for every layer.....

outp = []
print(len(check_pnts))
#
#load_trained

outs = {
    'conv_1': [],
    'conv_2': [],
    'conv_3': [],
    'conv_4': [],
}
conv_names = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

for chpnt in check_pnts:
    saver.restore(sess, chpnt)
    print("Loading ", chpnt, " Complete.")
    out_conv_1, out_conv_2, out_conv_3, out_conv_4 = sess.run([h3, h7, h11, h15], feed_dict={real_datas: checking_data})
    outs['conv_1'].append(np.transpose(out_conv_1, (0, 2, 3, 1)))
    outs['conv_2'].append(np.transpose(out_conv_2, (0, 2, 3, 1)))
    outs['conv_3'].append(np.transpose(out_conv_3, (0, 2, 3, 1)))
    outs['conv_4'].append(np.transpose(out_conv_4, (0, 2, 3, 1)))

from dft_ccas import fourier_ccas

num_ckpts = len(check_pnts)
num_conv = len(conv_names)

corr_coeffs = np.zeros((num_ckpts, num_conv, num_conv))

for i in range(num_ckpts):
    if i ==4:
        break
    for j in range(num_conv):
        
        layer_key_max = conv_names[j]
        out_train_max = outs[layer_key_max][-1]

        for k in range(num_conv):
            layer_key_curr = conv_names[k]
            out_train_curr = outs[layer_key_curr][i]
            
            # Compute CCA
            ccas = fourier_ccas(out_train_max, out_train_curr, return_coefs=True, compute_dirns=True)
            
            # Find number of dirs = min(num_neurons1, num_neurons2)
            num_direc = min(ccas['cca_coef1'][0].shape[0], ccas['cca_coef2'][0].shape[0])
            
            # Find mean corr coeff
            corr_coeffs[i][j][k] = np.mean(np.array([ccas['cca_coef1'][i][:num_direc] for i in range(ccas['cca_coef1'].values.shape[0])]))
            print('{},{},{} : Corr Coeff {}'.format(i,j,k, corr_coeffs[i,j,k]))

np.save('corr_coeffs.npy', corr_coeffs)