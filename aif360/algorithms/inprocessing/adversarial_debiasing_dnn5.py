import numpy as np


from aif360.load_model.util_functions import gradient_graph

try:
    import tensorflow.compat.v1 as tf
except ImportError as error:
    from logging import warning
    warning("{}: AdversarialDebiasing will be unavailable. To install, run:\n"
            "pip install 'aif360[AdversarialDebiasing]'".format(error))

from aif360.algorithms import Transformer
from tensorflow.python.platform import flags
from aif360.utils.utils_tf import *
from aif360.load_model.tutorial_models import dnn

FLAGS = flags.FLAGS

class AdversarialDebiasingDnn5(Transformer):
    """
    对DNN5模型的对抗训练去偏的实现
    """

    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 scope_name,
                 sess,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=1000,
                 batch_size=128,
                 classifier_num_hidden_units=200,
                 debias=True,
                 save = True):
        super(AdversarialDebiasingDnn5, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        self.scope_name = scope_name
        self.seed = seed

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

        self.sess = sess
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.save = save


        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None
        self.preds_symbolic_output = None


    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        计算分类器对outcome 变量（特征）的推理预测
        """
        with tf.variable_scope("classifier_model"):
            model = dnn(input_shape=(None, features_dim), nb_classes=1)
            dnn5 = model(features)
            self.preds_symbolic_output = dnn5
            pred_logit = model.get_logits(features)
            get_probs = model.get_probs(features)
            pred_label = get_probs
        return pred_label, pred_logit

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute.
        计算Adversary对于敏感属性的预测值
        """
        with tf.variable_scope("adversary_model", reuse=tf.AUTO_REUSE):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)
            # glorot函数需要查一下
            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed4))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

            return pred_protected_attribute_label, pred_protected_attribute_logit


    def save_model(self, train_dir, filename):
        if self.save:
            train_dir = os.path.join(train_dir, str(self.num_epochs - 1))
            if not os._exists(train_dir):
                os.makedirs(train_dir)
            save_path = os.path.join(train_dir, filename)
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)
            print("Completed model training and saved at: " +
                         str(save_path))
            self.model_path = save_path
        else:
            print("Completed model training.")


    def fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.
        利用梯度下降计算公平的分类器的模型参数-相当于整个去偏算法的主体部分
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if tf.executing_eagerly():
            # 在紧急执行的模式下，汇报运行时错误，因为对抗去偏不是即时工作的，需要在脚本开头加上关闭该模式的声明
            raise RuntimeError("AdversarialDebiasing does not work in eager "
                    "execution mode. To fix, add `tf.disable_eager_execution()`"
                    " to the top of the calling script.")

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()
        # dataset.labels应该是一个nparray 下列temp_labels应该是对所有的 所有在源标签中是fav的 temptabels数组的该行， 第1列数值赋值为1
        # 是unfav标签的行，的第一列处，赋值为0
        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        with tf.variable_scope(self.scope_name):
            # scopename在类构造时必须传入，个人理解表示是原来的分类器还是adversary的训练参数空间
            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Setup placeholders 设立tf的placeholder 相当于在建立计算的图
            # 依次是： 特征输入（分类器是x，adversary可能是y_, y,z或者单纯y_,z）
            # 敏感属性 维度是1
            # 真实标签 维度也是1
            # keep_prob dropout函数中的参数
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            # 获得分类器的推理和分类器的loss损失
            # 预测的标签， 预测的logits
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # debias是逻辑控制开关，表示是否是去偏模式
                # Obtain adversary predictions and adversary loss
                # 获得adversary的预测和loss
                # 输出一敏感属性标签
                # 输出二敏感属性logits是adver模型的输出，
                # 输入一 predlogits分类器带出来的logits，相当于y_
                # 输入二 真是的标签 即y
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                # 敏感属性的损失，tf的reducemean函数 内嵌一个交叉熵，输入标签是敏感属性的placeholder ，logits是预测的敏感属性logits即输出二
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph,
                                                            logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            # 设立优化器和学习率
            # 全局step 开始学习率 lr设置为按照指数衰减的函数进行变换
            # 优化器是Adam。传入上述学习率参数
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                # 如果开启对抗去偏模式，设置adversary的优化器，和分类器的设置一样
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            # 分类器的 变量保存
            classifier_vars = [var for var in tf.trainable_variables() if 'classifier_model' in var.name]
            if self.debias:
                # 如果是去偏模式，还要保存adver的参数
                adversary_vars = [var for var in tf.trainable_variables() if 'adversary_model' in var.name]
                # Update classifier parameters
                # 更新分类器的参数，首先要
                # 记录adver的梯度，由预测敏感属性的loss，和保存分类器的vars数组作为adver优化器计算梯度的函数的参数传入
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            # 这一步标准化 这里用了lambda表达式，即匿名函数，相当于传入了一个参数x的函数， 返回的是x / (tf.norm(x) + np.finfo(np.float32).tiny)
            # normalize是一个函数？！！确实，下面有调用
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            # 创立一个空列表来接受分类器的梯度
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                # 算法的理论核心 修改adver的grad值
                # g是分类器的梯度 h是adver的梯度
                if self.debias:
                    # 第一步 标准化 求出adver——grad的单位向量吗？unit 用来之后计算g在h上的投影
                    #  grad是我们的主角，因为是要用以更新W（分类器的参数）的，所以操作的g是分类器的梯度
                    # 第二部 减去proj_h g
                    # 第三步 减去可调项 a * grad(W) *Loss(A)
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                #     空列表保存原始的或者经过去偏算法计算的grad 和 var
                classifier_grads.append((grad, var))
            #     minimizer？损失最小化的意思吗 这一步应该是将自己优化的grad传给优化器
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                # 如果开了去偏算法的模式，那么需要更新adver的参数
                with tf.control_dependencies([classifier_minimizer]):
                    # 控制依赖的with语句，会先执行classifier_minimizer的操作，在执行下面的
                    # adver的minimizer ader的优化器 调用函数是minimize 输入预测的敏感属性目前的loss var列表为adver_vars,不需要加上全局step
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            # sess开始run，feed 如全局变量初始化器和局部变量初始化器
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            # 开始训练
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = dataset.features[batch_ids]
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1,1])
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)

                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, pred_labels_loss_value))
            if self.save:
                if self.debias:
                    self.save_model('../adebias-model-fix/adult/', 'test.model')
                else:
                    self.save_model('../org-model/adult-fix/', 'test.model')
        return self

    def my_normalize(self, x):
        return x / (tf.norm(x) + np.finfo(np.float32).tiny)

    def fit_without_train(self, dataset):
        if tf.executing_eagerly():
            raise RuntimeError("AdversarialDebiasing does not work in eager "
                    "execution mode. To fix, add `tf.disable_eager_execution()`"
                    " to the top of the calling script.")

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        temp_labels = dataset.labels.copy()
        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        with tf.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Setup placeholders 设立tf的placeholder 相当于在建立计算的图
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:

                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph,
                                                            logits=pred_protected_attributes_logits))

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables() if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables() if 'adversary_model' in var.name]
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            # normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):

                if self.debias:
                    unit_adversary_grad = self.my_normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]

                classifier_grads.append((grad, var))

            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

    def fit_and_pred(self, dataset, dataset_test, graph_dir=
                     '../org-model-merge/adult/graphlog'):
        self.fit(dataset)

        tf.train.write_graph(self.sess.graph_def, graph_dir, 'testmodel.pbtxt')

        dataset_new_train, dataset_new_test = self.predict(dataset), self.predict(dataset_test)
        return dataset_new_train, dataset_new_test


    def predict(self, dataset):
        num_test_samples, _ = np.shape(dataset.features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1, 1])
            batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                    dataset.protected_attribute_names.index(
                                                        self.protected_attribute_name)], [-1, 1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:, 0].tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        # 在这里 被提供的数据集的label被重新赋值了，可以基于此计算几个公平性指标，真的有点绕啊
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels) > 0.5).astype(np.float64).reshape(-1, 1)

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new

    def predict_with_load_gra(self, dataset):
        self.fit_without_train(dataset)
        if self.model_path is None:
            if self.debias:
                model_path = '../adebias-model/adult/999/test.model'
            else:
                model_path = '../org-model/adult/999/test.model'
        else:
            model_path = self.model_path
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(self.sess, model_path)

        num_test_samples, _ = np.shape(dataset.features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1,1])
            batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                         dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes
                            }

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy = True)
        # 在这里 被提供的数据集的label被重新赋值了，可以基于此计算几个公平性指标，真的有点绕啊
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels)>0.5).astype(np.float64).reshape(-1,1)


        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new
