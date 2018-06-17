#-*- coding: UTF-8 -*-
import tornado
from .Sentiment_lstm import *
import yaml
from keras.models import model_from_yaml
from tornado import ioloop
from tornado.httpserver import HTTPServer
from tornado.web import Application, RequestHandler

class MyyuqingHandler(RequestHandler):
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        self.render('yuqing.html',result='',sentence='')
    def post(self):
        try:
            string=self.get_argument(name='juzi')
        except:
            self.write('请输入句子')
        try:
            string = re_str(string)
            with open('../data/kehushuju.txt', 'a') as f:
                f.write(string+'\n')
            data = input_transform(string)
            result = model.predict_classes(data)
            if result[0][0] == 1:
                # print(string, ' 不敏感')
                result = '不敏感'
            else:
                # print(string, ' 敏感')
                result = '敏感'
            self.render('yuqing.html', result=result, sentence=string)
        except:
            self.write('请重新输入您的句子')

if __name__=='__main__':
    print('loading model......')
    with open('../终极lstm4_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print('loading weights......')
    model.load_weights('../终极lstm4_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('已经开始工作')
    app=Application(
        [(r'/',MyyuqingHandler),
         ],
        debug=True,template_path='templates',statics='statics'
    )
    server=HTTPServer(app)
    server.listen(8888)
    ioloop.IOLoop().current().start()
        # while True:
        # string=input('输入一句话:')