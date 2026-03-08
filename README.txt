介绍
src/(存放源代码)
	authority.py (pytorch官方transformer版本)
	test.py (自己构建的，无batch，无gpu版本)
	makedata.py (用于从data/cnm.txt中提取数据，包括中文句子，英文句子，词表)
model/(存储模型和优化器的保存，由于过大，没有上传)
	model.pth (模型文件)
	optimizer.pth (优化器文件)
data/(数据集)
	cmn.txt (数据原文本文件)
	w.pkl (词表，python list)
	
	
authority.py使用官方版本代码
	100%复制
	单个句子使用 "$" 和 "&" 作为起始和结束标识
	不使用分隔符号
	不使用unknown标识符
	提供了模型测试代码
		模式1从每个batch中提取第一个句子测试
		模式2使用自回归生成，先给出 "$" ，再循环生成
test.py使用自建代码
	单个句子使用 "$" 和 "&" 作为起始和结束标识
	不使用分隔符号
	不使用unknown标识符
	不使用padding统一句子长度
