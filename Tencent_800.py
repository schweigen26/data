#-*- coding=utf8 -*-
from flask import Flask,jsonify,Response
from gensim.models import KeyedVectors
import json

server = Flask(__name__)
server.config['JSON_AS_ASCII'] = False


file = "Tencent_AILab_ChineseEmbedding.txt"
wv_from_text = KeyedVectors.load_word2vec_format(file,binary=False)
#wv_from_text = KeyedVectors.load(file,mmap='r+')
wv_from_text.init_sims(replace=True)

@server.route('/informationGet/<word>')
def informationGet(word):
    if word in wv_from_text.wv.vocab.keys():
        vec = wv_from_text[word]
        #return jsonify(wv_from_text.most_similar(positive=[vec],topn=20))
        return Response(json.dumps(wv_from_text.most_similar(positive=[vec],topn=20),indent=1,ensure_ascii=False),mimetype='application/json')
    else:
        return "没找到"

@server.route('/')
def index():

    return "加载成功"

if __name__ == '__main__':
    server.run(host='192.168.131.110',port=12315,debug=False)
