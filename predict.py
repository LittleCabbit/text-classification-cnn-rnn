# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['销售订单：     业务类型:分期收款     单据号：XS000473     订单日期：2015-06-18     销售类型：批发销售     客户简称：银嘉科技有限公司     存货名称：女士普通太阳眼镜     数量:1719     单价：382     不含税总价：656658     税率：0.17     总税额：111631     价税合计：768289     预发货日期：2015-06-18  发货单：     业务类型:分期收款     单据号：FH000473     发货日期：2015-06-18     销售类型：批发销售     客户简称：银嘉科技有限公司     存货名称：女士普通太阳眼镜     数量:1719     单价：382     不含税总价：656658     税率：0.17     总税额：111631     价税合计：768289      订单号：XS000473  出库单：     业务类型:分期收款     单据号：CK000473     出库日期：2015-06-18     销售类型：批发销售     客户简称：银嘉科技有限公司     存货名称：女士普通太阳眼镜     数量:1719     单价：382     不含税总价：656658  北京增值税专用发票     发票编号：8237723     开票日期:2015-06-18      购买方：      名称：银嘉科技有限公司     纳税人识别号：427889042193     地址、电话：河南省武汉市龙潭黄街q座 763286, 电话：010-41905329     开户行及账号：中国工商银行北京市朝阳支行648196321492432394          货物名称：女士普通太阳眼镜     数量：1031     单价：382     金额:393994     税率:0.17     税额：66979     合计：460973       购买方：     名称：北京亮康眼镜公司     纳税人识别号：1101082121202     地 址、电 话：北京市昌平区昌平路78号，电话：010-60228226     开户行及账号：中国工商银行北京市昌平支行1102020526782987908   北京增值税专用发票     发票编号：8237724     开票日期:2015-06-19      购买方：      名称：银嘉科技有限公司     纳税人识别号：427889042193     地址、电话：河南省武汉市龙潭黄街q座 763286, 电话：010-41905329     开户行及账号：中国工商银行北京市朝阳支行648196321492432394          货物名称：女士普通太阳眼镜     数量：687     单价：382     金额:262663     税率:0.17     税额：44652     合计：307315       购买方：     名称：北京亮康眼镜公司     纳税人识别号：1101082121202     地 址、电 话：北京市昌平区昌平路78号，电话：010-60228226     开户行及账号：中国工商银行北京市昌平支行1102020526782987908   合同编号：XS000473 ,    卖方：北京亮康眼镜公司     买方:银嘉科技有限公司     货物的名称：女士普通太阳眼镜     数量：1719     单价（不含税）：382     税率：0.17     价税合计：768289     付款方式：转账支票     付款时间:分两次开票和收款。签订合同当日，买方向卖方支付六成货款，第二天支付剩余货款     发货方式:运输费用由买方承担     日期：2015-06-18 ',
                 '销售订单：     业务类型:普通销售     单据号：XS000474     订单日期：2010-04-14     销售类型：批发销售     客户简称：创联世纪科技有限公司     存货名称：女士普通太阳眼镜     数量:1908     单价：288     不含税总价：549504     税率：0.17     总税额：93415     价税合计：642919     预发货日期：2010-04-14  发货单：     业务类型:普通销售     单据号：FH000474     发货日期：2010-04-14     销售类型：批发销售     客户简称：创联世纪科技有限公司     存货名称：女士普通太阳眼镜     数量:1908     单价：288     不含税总价：549504     税率：0.17     总税额：93415     价税合计：642919      订单号：XS000474  出库单：     业务类型:普通销售     单据号：CK000474     出库日期：2010-04-14     销售类型：批发销售     客户简称：创联世纪科技有限公司     存货名称：女士普通太阳眼镜     数量:1908     单价：288     不含税总价：549504  北京增值税专用发票     发票编号：8638909     开票日期:2010-04-14      购买方：      名称：创联世纪科技有限公司     纳税人识别号：750196821725     地址、电话：内蒙古自治区琴市南湖李路M座 978451, 电话：010-94031744     开户行及账号：中国工商银行北京市朝阳支行223364040751095859          货物名称：女士普通太阳眼镜     数量：1908     单价：288     金额:549504     税率:0.17     税额：93415     合计：642919       购买方：     名称：北京亮康眼镜公司     纳税人识别号：1101082121202     地 址、电 话：北京市昌平区昌平路78号，电话：010-60228226     开户行及账号：中国工商银行北京市昌平支行1102020526782987908   合同编号：XS000474 ,    卖方：北京亮康眼镜公司     买方:创联世纪科技有限公司     货物的名称：女士普通太阳眼镜     数量：1908     单价（不含税）：288     税率：0.17     价税合计：642919     付款方式：转账支票     付款时间:签订合同当日，买方向卖方支付全部货款     发货方式:运输费用由买方承担     日期：2010-04-14 ']
    for i in test_demo:
        print(cnn_model.predict(i))
