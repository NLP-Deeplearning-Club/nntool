Search.setIndex({docnames:["index","modules","nntool","nntool.abc","nntool.activation_functionabc","nntool.layer","nntool.model","nntool.neuron","nntool.objective","nntool.objective.loss_function","nntool.objective.regularizer","nntool.optimizer","nntool.utils"],envversion:53,filenames:["index.rst","modules.rst","nntool.rst","nntool.abc.rst","nntool.activation_functionabc.rst","nntool.layer.rst","nntool.model.rst","nntool.neuron.rst","nntool.objective.rst","nntool.objective.loss_function.rst","nntool.objective.regularizer.rst","nntool.optimizer.rst","nntool.utils.rst"],objects:{"":{nntool:[2,0,0,"-"]},"nntool.abc":{functionabc:[3,0,0,"-"],layerabc:[3,0,0,"-"],modelabc:[3,0,0,"-"],neuronabc:[3,0,0,"-"]},"nntool.abc.functionabc":{ActivationFunctionABC:[3,1,1,""],FunctionABC:[3,1,1,""],LossFuction:[3,1,1,""],NormABC:[3,1,1,""]},"nntool.abc.functionabc.FunctionABC":{d:[3,2,1,""]},"nntool.abc.layerabc":{ActivationFunctionLayer:[3,1,1,""],HiddenLayer:[3,1,1,""],LayerABC:[3,1,1,""],NeuronLayer:[3,1,1,""]},"nntool.abc.layerabc.HiddenLayer":{size:[3,3,1,""]},"nntool.abc.layerabc.LayerABC":{backward:[3,2,1,""],djdys:[3,3,1,""],forward:[3,2,1,""],x:[3,3,1,""],y:[3,3,1,""]},"nntool.abc.layerabc.NeuronLayer":{Thetas:[3,3,1,""],djdThetas:[3,3,1,""],djdxs:[3,3,1,""],input_size:[3,3,1,""],size:[3,3,1,""],update_Theta:[3,2,1,""]},"nntool.abc.modelabc":{ModelABC:[3,1,1,""]},"nntool.abc.modelabc.ModelABC":{add:[3,2,1,""],fit:[3,2,1,""],predict:[3,2,1,""],train:[3,2,1,""],trained:[3,3,1,""]},"nntool.abc.neuronabc":{NeuronABC:[3,1,1,""]},"nntool.activation_functionabc":{embedding_lookup:[4,0,0,"-"],reLu:[4,0,0,"-"],sigmoid:[4,0,0,"-"],tanh:[4,0,0,"-"]},"nntool.activation_functionabc.embedding_lookup":{EmbeddingLookup:[4,1,1,""]},"nntool.activation_functionabc.embedding_lookup.EmbeddingLookup":{d:[4,2,1,""],d_Theta:[4,2,1,""],d_x:[4,2,1,""]},"nntool.activation_functionabc.reLu":{ReLu:[4,1,1,""]},"nntool.activation_functionabc.reLu.ReLu":{d:[4,2,1,""],d_Theta:[4,2,1,""],d_x:[4,2,1,""]},"nntool.activation_functionabc.sigmoid":{Sigmoid:[4,1,1,""]},"nntool.activation_functionabc.sigmoid.Sigmoid":{d:[4,2,1,""],d_Theta:[4,2,1,""],d_x:[4,2,1,""]},"nntool.activation_functionabc.tanh":{Tanh:[4,1,1,""]},"nntool.activation_functionabc.tanh.Tanh":{d:[4,2,1,""],d_Theta:[4,2,1,""],d_x:[4,2,1,""]},"nntool.layer":{embedding_layer:[5,0,0,"-"],linear_layer:[5,0,0,"-"],sigmoid_layer:[5,0,0,"-"],softmax_layer:[5,0,0,"-"],tanh_layer:[5,0,0,"-"]},"nntool.layer.embedding_layer":{EmbeddingLayer:[5,1,1,""]},"nntool.layer.embedding_layer.EmbeddingLayer":{Theta:[5,3,1,""],backward:[5,2,1,""],d:[5,2,1,""],d_Theta:[5,2,1,""],d_x:[5,2,1,""],djdys:[5,3,1,""],forward:[5,2,1,""],size:[5,3,1,""],x:[5,3,1,""],y:[5,3,1,""]},"nntool.layer.linear_layer":{LinearNeuronLayer:[5,1,1,""]},"nntool.layer.linear_layer.LinearNeuronLayer":{backward:[5,2,1,""],forward:[5,2,1,""],size:[5,3,1,""],update_Theta:[5,2,1,""]},"nntool.layer.sigmoid_layer":{SigmoidLayer:[5,1,1,""]},"nntool.layer.sigmoid_layer.SigmoidLayer":{backward:[5,2,1,""],d_x:[5,2,1,""],forward:[5,2,1,""]},"nntool.layer.softmax_layer":{SoftmaxLayer:[5,1,1,""]},"nntool.layer.softmax_layer.SoftmaxLayer":{backward:[5,2,1,""],d:[5,2,1,""],forward:[5,2,1,""],size:[5,3,1,""]},"nntool.layer.tanh_layer":{TanhLayer:[5,1,1,""]},"nntool.layer.tanh_layer.TanhLayer":{backward:[5,2,1,""],d:[5,2,1,""],d_x:[5,2,1,""],forward:[5,2,1,""]},"nntool.model":{sequential:[6,0,0,"-"]},"nntool.model.sequential":{Sequential:[6,1,1,""]},"nntool.model.sequential.Sequential":{add:[6,2,1,""],fit:[6,2,1,""],predict:[6,2,1,""],train:[6,2,1,""],trained:[6,3,1,""]},"nntool.neuron":{linear_neuron:[7,0,0,"-"]},"nntool.neuron.linear_neuron":{LinearNeuron:[7,1,1,""]},"nntool.neuron.linear_neuron.LinearNeuron":{d:[7,2,1,""],d_Theta:[7,2,1,""],d_x:[7,2,1,""],update_Theta:[7,2,1,""]},"nntool.objective":{loss_function:[9,0,0,"-"],regularizer:[10,0,0,"-"]},"nntool.objective.loss_function":{corss_entropy:[9,0,0,"-"],hinge:[9,0,0,"-"],mle:[9,0,0,"-"],mse:[9,0,0,"-"],square_loss:[9,0,0,"-"]},"nntool.objective.loss_function.corss_entropy":{CrossEntropy:[9,1,1,""]},"nntool.objective.loss_function.corss_entropy.CrossEntropy":{backward:[9,2,1,""],d:[9,2,1,""]},"nntool.objective.loss_function.hinge":{hinge:[9,4,1,""]},"nntool.objective.loss_function.mle":{mle:[9,4,1,""]},"nntool.objective.loss_function.mse":{MSE:[9,1,1,""]},"nntool.objective.loss_function.mse.MSE":{dx:[9,2,1,""]},"nntool.objective.loss_function.square_loss":{square_loss:[9,4,1,""]},"nntool.objective.regularizer":{l2:[10,0,0,"-"]},"nntool.objective.regularizer.l2":{L2_Norm:[10,1,1,""]},"nntool.objective.regularizer.l2.L2_Norm":{dx:[10,2,1,""]},"nntool.optimizer":{BGD:[11,0,0,"-"]},"nntool.optimizer.BGD":{BGDRuner:[11,1,1,""]},"nntool.optimizer.BGD.BGDRuner":{backward:[11,2,1,""],forward:[11,2,1,""]},"nntool.utils":{init_factory:[12,0,0,"-"]},"nntool.utils.init_factory":{normal_factory:[12,4,1,""],uniform_factory:[12,4,1,""]},nntool:{abc:[3,0,0,"-"],activation_functionabc:[4,0,0,"-"],layer:[5,0,0,"-"],model:[6,0,0,"-"],neuron:[7,0,0,"-"],objective:[8,0,0,"-"],optimizer:[11,0,0,"-"],utils:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"\u4e00\u822c\u63a5\u5728\u7ebf\u6027\u5c42\u540e\u9762\u4f7f\u7528":3,"\u4e00\u822c\u7528\u4e8e\u505a\u6324\u538b":5,"\u4e00\u822c\u914d\u5408softmax\u4e00\u8d77\u4f7f\u7528":9,"\u4e0d\u5b9e\u73b0":9,"\u4e14\u5404\u9879\u76f8\u52a0\u548c\u4e3a1\u7684\u4e0e\u8f93\u5165\u5411\u91cf\u7b49\u957f\u7684\u5411\u91cf":5,"\u4e5f\u53ef\u4ee5\u8bf4\u662f\u5168\u8fde\u63a5\u5c42":5,"\u4e5f\u5c31\u662f\u611f\u77e5\u5668":7,"\u4e5f\u5c31\u662fw":7,"\u4e5f\u9700\u8981\u53ef\u4ee5\u9006\u5411\u6c42\u51fa\u5bf9x\u548c\u5bf9\u53c2\u6570theta\u7684\u504f\u5bfc":3,"\u4ea4\u53c9\u71b5\u635f\u5931\u51fd\u6570":9,"\u4ea4\u53c9\u71b5\u73b0\u5728\u53ea\u548csoftmaxt\u914d\u5408\u4f7f\u7528":9,"\u4f5c\u7528\u662f\u5c06\u7ebf\u6027\u5c42\u4f20\u5165\u7684\u5185\u5bb9\u6620\u5c04\u5230":5,"\u503c\u57df\u5728":5,"\u505a":5,"\u505a\u5f00\u5173":5,"\u505a\u6982\u7387":5,"\u5176":[],"\u5176\u4e24\u7aef\u5e73\u7f13\u4e2d\u95f4\u9661\u5ced":5,"\u518d\u8f93\u51fa\u5176\u7ed3\u679c":3,"\u521d\u59cb\u5316\u5de5\u5177":12,"\u524d\u5411\u8fd0\u7b97":11,"\u533a\u95f4":5,"\u53cc\u66f2\u6b63\u5f26\u51fd\u6570":5,"\u53cd\u5411\u4f20\u64ad\u7b97\u6cd5\u7684\u7ed3\u679c":3,"\u53cd\u5411\u8fd0\u7b97":11,"\u540c\u65f6\u964d\u4f4e\u7ef4\u5ea6":5,"\u56de\u5f52":9,"\u56e0\u4e3a\u7ebf\u6027\u6a21\u578b\u7684\u8868\u8fbe\u80fd\u529b\u4e0d\u591f":3,"\u56e0\u6b64\u5728\u4e24\u7aef\u5bb9\u6613\u51fa\u73b0\u68af\u5ea6\u6d88\u5931\u7684\u60c5\u51b5":5,"\u5747\u65b9\u5dee\u635f\u5931\u51fd\u6570":9,"\u5bf9theta\u548c\u5bf9x\u7684\u504f\u5bfc":3,"\u5bf9theta\u7684\u504f\u5bfc":[4,5,7],"\u5bf9x\u7684\u504f\u5bfc":5,"\u5c06\u6a21\u578b\u7406\u89e3\u4e3a\u5c42\u7684\u5806\u53e0":6,"\u5c42\u4e5f\u6709\u7279\u6b8a\u7684\u5730\u65b9":3,"\u5c42\u8fd9\u662f\u4e00\u4e2a\u62bd\u8c61\u6982\u5ff5":3,"\u5d4c\u5165\u5c42":5,"\u5e38\u7528\u4e8e\u6700\u5927\u5316\u5229\u6da6":9,"\u5e38\u7528\u4e8ennl":9,"\u5e38\u7528\u5728svm":9,"\u5e38\u89c1\u4e8elogist":9,"\u5e73\u65b9\u635f\u5931\u51fd\u6570":9,"\u5e76\u5c06\u5176\u4fdd\u5b58\u4e3a":5,"\u5e76\u5c06\u5176\u4fdd\u5b58\u4e3aself":3,"\u5e8f\u5217\u6a21\u578b":6,"\u5f53\u591a\u4e2a\u7ed3\u679c\u8981\u8fdb\u884c\u76f8\u540c\u6216\u8005\u76f8\u5173\u64cd\u4f5c\u7684\u65f6\u5019":3,"\u5f62\u72b6\u4e0esigmoid\u51fd\u6570\u7c7b\u4f3c":5,"\u6324\u538b":5,"\u635f\u5931\u51fd\u6570":3,"\u662f\u5426\u5df2\u7ecf\u8bad\u7ec3\u8fc7":[3,6],"\u66f4\u65b0\u53c2\u6570":3,"\u6700\u540e\u4e00\u5c42\u7684hat_y\u5373\u4e3a\u9884\u6d4b\u503c":2,"\u6709\u7684\u53ef\u80fd\u6ca1\u6709\u5176\u4e2d\u67d0\u4e00\u9879\u4f46\u8fd9\u4e0d\u91cd\u8981":3,"\u6709\u7684\u6709\u53c2\u6570\u7684\u5c31\u9700\u8981\u6bcf\u6b21\u8bad\u7ec3\u66f4\u65b0\u53c2\u6570":3,"\u672c\u5904\u7684\u5b9e\u73b0\u4e3amin":11,"\u672c\u5c42\u7684\u7eac\u5ea6":5,"\u672c\u5c42\u7684\u8f93\u51fa\u7eac\u5ea6":[3,5],"\u672c\u6a21\u5757\u4e2d":2,"\u68af\u5ea6\u4e0b\u964d":11,"\u68af\u5ea6\u4e0b\u964d\u6cd5":11,"\u6a21\u578b\u662f\u7531\u5c42\u7ec4\u5408\u800c\u6210\u7684\u8fd0\u7b97\u7ed3\u6784":3,"\u6b63\u5219\u5316\u53c2\u6570\u7b49\u4ef7\u4e8e\u5bf9\u53c2\u6570\u5f15\u5165\u5148\u9a8c\u5206\u5e03":3,"\u6b63\u5219\u9879\u51fd\u6570":3,"\u6bd4\u5982\u5e26\u6709\u8d85\u53c2\u6570\u7684\u7684\u5c31\u8981\u8bbe\u5b9a\u8d85\u53c2\u6570":3,"\u6d4b\u8bd5\u5728dev\u6570\u636e\u96c6\u4e0a\u7684\u6548\u679c":6,"\u6dfb\u52a0\u4e00\u5c42":3,"\u6fc0\u6d3b\u51fd\u6570\u4e00\u822c\u662f\u7528\u6765\u52a0\u5165\u975e\u7ebf\u6027\u56e0\u7d20\u7684":3,"\u6fc0\u6d3b\u51fd\u6570\u5c42\u4e00\u822c\u53ea\u662f\u5c06\u4e0a\u4e00\u5c42\u4f20\u5165\u7684\u8f93\u5165\u8fdb\u884c\u975e\u7ebf\u6027\u53d8\u6362":3,"\u7528\u4e8e\u53cd\u5411\u4f20\u64ad\u8ba1\u7b97\u795e\u7ecf\u5143\u53c2\u6570":3,"\u7528\u4e8e\u76f4\u89c2\u8bc4\u4ef7\u6a21\u578b\u9884\u8ba1\u503c\u4e0e\u771f\u5b9e\u503c\u95f4\u7684\u5dee\u8ddd":3,"\u7528\u4e8e\u9632\u6b62\u8fc7\u62df\u5408":3,"\u7684\u6fc0\u6d3b\u51fd\u6570":5,"\u795e\u7ecf\u5143\u4e00\u822c\u90fd\u6709\u53c2\u6570":3,"\u795e\u7ecf\u5143\u5176\u5b9e\u4e5f\u662f\u4e00\u4e2a\u51fd\u6570":3,"\u795e\u7ecf\u5143\u5c42":3,"\u795e\u7ecf\u7f51\u7edc\u4e2d\u6a21\u578b\u57fa\u672c\u4e0a\u5c31\u662f\u82b1\u5f0f\u5806\u53e0\u5c42":3,"\u795e\u7ecf\u7f51\u7edc\u4e2d\u7684\u51fd\u6570\u591a\u662f\u8fdb\u884c\u77e9\u9635\u8ba1\u7b97":3,"\u795e\u7ecf\u7f51\u7edc\u7b97\u6cd5\u672c\u8d28\u4e0a\u5c31\u662f\u8bad\u7ec3\u5404\u4e2a\u795e\u7ecf\u5143\u7684\u53c2\u6570":3,"\u7ebf\u6027\u795e\u7ecf\u5143":7,"\u7ebf\u6027\u795e\u7ecf\u5143\u5c42":5,"\u800c\u8bad\u7ec3\u8fd9\u4e9b\u53c2\u6570\u4e5f\u5c31\u662f\u6211\u4eec\u7684\u4efb\u52a1":3,"\u800cy\u6307\u4ee3\u6807\u7b7e\u6570\u636e":2,"\u8868\u793a\u672c\u5c42\u90fd\u662f\u795e\u7ecf\u5143":3,"\u8ba1\u7b97\u672c\u5c42\u6b63\u5411\u8f93\u51fa":3,"\u8ba1\u7b97\u672c\u5c42\u8f93\u51fa":[3,5],"\u8ba1\u7b97\u6a21\u578b\u7684\u6b63\u5411\u8ba1\u7b97\u7ed3\u679c":[3,5],"\u8bad\u7ec3\u6a21\u578b":6,"\u8f6c\u4e3a\u7a20\u5bc6\u77e9\u9635":5,"\u8f93\u5165\u7684\u6bcf\u4e00\u4e2a\u7eac\u5ea6\u90fd\u4f1a\u8fdb\u5165\u5c42\u5185\u7684\u6bcf\u4e2a\u795e\u7ecf\u5143\u8fdb\u884c\u8ba1\u7b97":3,"\u8fd9\u4e2a\u5411\u91cf\u53ef\u4ee5\u4f5c\u4e3a\u591a\u503c\u5206\u7c7b\u7684\u5404\u4f4d\u5bf9\u5e94\u7684\u503c\u7684\u53ef\u80fd\u6027":5,"\u8fd9\u4e2a\u662fkeras\u4e2d\u7684\u6982\u5ff5":6,"\u8fd9\u5c31\u53ef\u4ee5\u88ab\u770b\u4f5c\u662f\u4e00\u5c42":3,"\u8fd9\u5c42\u4e00\u822c\u6765\u8bf4\u662f\u7528\u4f5c\u5c06\u7a00\u758f\u7684\u8f93\u5165\u77e9\u9635":5,"\u9690\u85cf\u5c42\u662f\u6307\u9664\u53bb\u8f93\u5165\u8f93\u51fa\u7684\u6240\u6709\u5c42":3,"\u9700\u8981\u4e0e\u5176\u8f93\u51fa\u4e3a":9,"\u9700\u8981\u53ef\u4ee5\u6b63\u5411\u6c42\u51fa\u503c":3,"\u9884\u6d4b\u4e00\u7ec4\u6570\u636e":[3,6],"b\u4e3a\u504f\u7f6e":7,"class":[3,4,5,6,7,9,10,11],"function":5,"hat_y\u6307\u4ee3\u524d\u5411\u4f20\u5230\u6b65\u9aa4\u7684\u8ba1\u7b97\u7ed3\u679c":2,"int":[],"l2\u6b63\u5219\u9879":10,"log\u5bf9\u6570\u635f\u5931\u51fd\u6570":9,"logistic\u6fc0\u6d3b\u51fd\u6570\u5c42":5,"onehot\u7f16\u7801":5,"sigmoid\u7684\u7528\u9014\u4e00\u822c\u53ef\u4ee5\u5f52\u7ed3\u4e3a":5,"softmax\u51fd\u6570\u7528\u4e8e\u5c06\u8f93\u5165\u7684\u5411\u91cf\u8f6c\u6362\u4e3a\u6bcf\u9879\u503c\u57df\u5728":5,"w\u4e3a\u5bf9\u5e94\u8f93\u5165\u7684\u6743\u91cd":7,"x\u6307\u4ee3\u5411\u91cf\u7684\u8f93\u5165":2,"x\u6307\u4ee3\u77e9\u9635\u7684\u8f93\u5165":2,_warp:5,abc:[0,1,2,4,5,6,7,9,10],activation_functionabc:[0,1,2],activationfunctionabc:3,activationfunctionlay:[3,5],add:[3,6],api:0,backward:[3,5,9,11],base:[3,4,5,6,7,9,10,11],batch_siz:11,bgd:[1,2],bgdruner:11,content:[0,1],corss_entropi:[2,8],crossentropi:9,d_theta:[4,5,7],d_x:[4,5,7],dev:3,djdtheta:3,djdx:3,djdy:[3,4,5,7,9],embedding_lay:[1,2],embedding_lookup:[1,2],embeddinglay:5,embeddinglookup:4,epoch:11,eta:[3,5,9,11],fit:[3,6],forward:[3,5,11],functionabc:[1,2,4,9,10],hiddenlay:[3,5],hing:[2,8],index:0,init_factori:[1,2,5],input_s:3,l2_norm:10,lambd:10,layer:[0,1,2,3],layerabc:[1,2,5],linear_lay:[1,2],linear_neuron:[1,2],linearneuron:7,linearneuronlay:5,local:5,loss_funct:[2,8],lossfuct:3,math:[],mle:[2,8],model:[0,1,2],modelabc:[1,2,6],modul:[0,1],mse:[2,8],neuron:[0,1,2],neuronabc:[1,2,7],neuronlay:[3,5],new_theta:7,none:[3,5,11],normabc:3,normal_factori:12,object:[0,1,2,11],optim:[0,1,2],out:9,packag:[0,1],page:0,partial:[],predict:[3,6],rac:[],range_:12,regular:[2,8],relu:[1,2],search:0,sequenti:[1,2],sigmoid:[1,2],sigmoid_lay:[1,2],sigmoidlay:5,size:[3,5],softmax:[],softmax_lay:[1,2],softmaxlay:5,sourc:[3,4,5,6,7,9,10,11,12],square_loss:[2,8],submodul:[1,2,8],subpackag:[0,1],sum_:[],tanh:[1,2],tanh_lay:[1,2],tanhlay:5,theta:[3,4,5,7],train:[3,6],trainner:6,uniform_factori:[5,12],update_theta:[3,5,7],util:[0,1,2],x_i:[],x_j:[],x_test:[3,6],y_i:[],y_iy_j:[]},titles:["Welcome to nntool\u2019s documentation!","nntool","nntool package","nntool.abc package","nntool.activation_functionabc package","nntool.layer package","nntool.model package","nntool.neuron package","nntool.objective package","nntool.objective.loss_function package","nntool.objective.regularizer package","nntool.optimizer package","nntool.utils package"],titleterms:{abc:3,activation_functionabc:4,bgd:11,content:[2,3,4,5,6,7,8,9,10,11,12],corss_entropi:9,document:0,embedding_lay:5,embedding_lookup:4,functionabc:3,hing:9,indic:0,init_factori:12,layer:5,layerabc:3,linear_lay:5,linear_neuron:7,loss_funct:9,mle:9,model:6,modelabc:3,modul:[2,3,4,5,6,7,8,9,10,11,12],mse:9,neuron:7,neuronabc:3,nntool:[0,1,2,3,4,5,6,7,8,9,10,11,12],object:[8,9,10],optim:11,packag:[2,3,4,5,6,7,8,9,10,11,12],regular:10,relu:4,sequenti:6,sigmoid:4,sigmoid_lay:5,softmax_lay:5,square_loss:9,submodul:[3,4,5,6,7,9,10,11,12],subpackag:[2,8],tabl:0,tanh:4,tanh_lay:5,util:12,welcom:0}})