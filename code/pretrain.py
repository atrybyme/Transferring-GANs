import sys, locale
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
tf.app.flags.DEFINE_string("sResultTag", "class_v0", "your tag for each test case")

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", False, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/", "where to save the checkpoint and sample")

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
tf.app.flags.DEFINE_integer("iFilterDimsD", 64, "")

tf.app.flags.DEFINE_float("fDropRate", 0.3, "")

cfg(sys.argv)

allocate_gpu()

############################################################################################################################################

def discriminator_dcgan(input, num_logits):

    iFilterDimsD = cfg.iFilterDimsD

    with tf.variable_scope('discriminator', tf.AUTO_REUSE):

        h0 = input
        h0 = noise(h0, 0.1, bAdd=True)

        h0 = conv2d(h0, iFilterDimsD * 1, ksize=3, stride=1, name='conv32')  # 32x32
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 2, ksize=3, stride=2, name='conv32_16')  # 32x32 --> 16x16
        h0 = batch_norm(h0, name='bn16')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 4, ksize=3, stride=2, name='conv16_8')  # 16x16 --> 8x8
        h0 = batch_norm(h0, name='bn8')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = conv2d(h0, iFilterDimsD * 8, ksize=3, stride=2, name='conv8_4')  # 8x8 --> 4x4
        h0 = batch_norm(h0, name='bn4')
        h0 = activate(h0, cfg.oAct)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = avgpool(h0, h0.get_shape().as_list()[2], h0.get_shape().as_list()[3])
        h0 = tf.contrib.layers.flatten(h0)
        h0 = dropout(h0, cfg.fDropRate)

        h0 = linear(h0, num_logits)

        return h0

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
data_gen = labeled_data_gen_epoch(dataX, dataY, cfg.iBatchSize)

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag) else "") + cfg.sDataSet

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

discriminator = discriminator_dcgan

real_datas = tf.placeholder(tf.float32, [None, cfg.iDimsC, 32, 32], name='real_datas')
real_labels = tf.placeholder(tf.int32, shape=[None])

num_logits = 1000
real_logits = discriminator(real_datas, num_logits)

dis_total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_logits, labels=real_labels))

prediction = tf.cast(tf.argmax(real_logits, 1), tf.int32)
equality = tf.equal(prediction, real_labels)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')

lr = cfg.fLrIni * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))


dis_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
dis_gradient_values = dis_optimizer.compute_gradients(dis_total_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values, global_step=global_step)

saver = tf.train.Saver(max_to_keep=1000)

############################################################################################################################################

iter = 0
last_save_time = last_log_time = last_plot_time = last_score_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n\n')
            logger.flush()
            logger.log('\n\n')
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

logger.log("Discriminator Total Parameter Count: {}".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

while iter <= cfg.iMaxIter:

    iter += 1
    start_time = time.time()

    _datas, _labels = data_gen.__next__()
    _, _dis_total_loss, _lr, _acc = sess.run(
        [dis_optimize_ops, dis_total_loss, lr, accuracy],
        feed_dict={real_datas: _datas, real_labels: _labels})

    logger.tick(iter)
    logger.info('klr', _lr * 1000)
    logger.info('acc', _acc * 100)
    logger.info('time', time.time() - start_time)

    logger.info('loss_dis_gan', _dis_total_loss)

    if time.time() - last_save_time > 60*5:
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()
        logger.log('Model Saved\n\n')

    logger.flush()
    last_log_time = time.time()