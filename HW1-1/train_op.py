import tensorflow as tf 
import os
import numpy as np
EPOCH = 10000
BATCH_SIZE = 200
def train_op(train_x,train_y,model,dir_name):
	tf.reset_default_graph()
	graph = tf.Graph()
	loss_temp_record = []
	with graph.as_default():
		x = tf.placeholder(tf.float32,(None,1))
		print(x.name)
		y = tf.placeholder(tf.float32,(None,1))
		#global_step = tf.Variable(0,trainable=False)
		#shallow_model = Shallow()
		y_pred = model(x)
		global_step = tf.Variable(0,trainable=False)
		print(y_pred.name)
		loss = tf.reduce_mean((y-y_pred)**2)
		tf.summary.scalar('loss',loss)

		train_step = tf.train.AdamOptimizer(learning_rate=0.0005,beta1=0.5).minimize(loss,global_step=global_step)
		summary_op = tf.summary.merge_all()
		tmp_dir='./tmp/'
		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)
		summary_writer = tf.summary.FileWriter(tmp_dir,graph)
		saver = tf.train.Saver()
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
	with tf.Session(graph=graph) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		ckpt = tf.train.get_checkpoint_state(dir_name)
		if ckpt and ckpt.model_checkpoint_path:
			print(" Model has been exists")
			return y_pred.name,0
		for epoch in range(1,EPOCH+1):
			for batch in range(train_x.shape[0]//BATCH_SIZE):
				x_batch  = np.reshape(train_x[batch:batch+BATCH_SIZE],(BATCH_SIZE,1))
				y_batch  = np.reshape(train_y[batch:batch+BATCH_SIZE],(BATCH_SIZE,1))
				_,loss_value,step = sess.run([train_step,loss,global_step],feed_dict={x:x_batch,y:y_batch})

				#print(step)
			loss_temp_record.append(loss_value)
			print("epoch:{}/{},loss:{}.\n".format(epoch,EPOCH,loss_value))
			summary = sess.run(summary_op,feed_dict={x:x_batch,y:y_batch})
			summary_writer.add_summary(summary,epoch)
			ckpt_dir = os.path.join(dir_name,'model')
			saver.save(sess,ckpt_dir,global_step=global_step)
	return y_pred.name,loss_temp_record