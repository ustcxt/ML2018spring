import tensorflow as tf
import numpy as np
def test_op(x,model,ckpt_dir,op_name):
	tf.reset_default_graph()
	graph = tf.Graph()
	f = open(ckpt_dir+'/checkpoint','r')
	line = f.readline()
	#print(line.split(':')[-1].split('-')[-1].split('"')[0])
	num=line.split(':')[-1].split('-')[-1].split('"')[0]

	with tf.Session(graph=graph) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		new_saver = tf.train.import_meta_graph(ckpt_dir+'/model-'+num+'.meta')

		new_saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
		x_t = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
		pred = tf.get_default_graph().get_tensor_by_name(op_name)
		y_pred = sess.run(pred,feed_dict={x_t:x})
		#y_pred = sess.run(y_pred,feed_dict={x:x})
	return y_pred

