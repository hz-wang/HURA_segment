# HURA_segment

### Bing数据抽取及预处理

#### 1 网站选取及处理

##### 1.1 选定50个关于某个domain的website

##### 1.2 将`domain_website.txt`的文件上传至`COSMOS`的`/my`文件夹下

#### 2 抽取用户及处理

##### 2.1 修改COSMOS上的`ExtractPositive.script`程序中对应输出和资源的文件名分别为`domain_positive_user_time.txt`和`domain_website.txt`,  `ExtractPositive.script.cs` 程序中对应的文件名为`domain_website.txt`

##### 2.2 修改`ExtractPositive.script`程序中的`click_count`参数，当其为1时保存为`domain_positive_click_user.txt`，当其为0是保存为`domain_positive_search_user.txt`

##### 2.3 运行`ExtractPositive.script`程序获取该domain下的`positive user`(包括click_user和search_user)

##### 2.4 将该`domain`下的正向用户下载到本地，使用`calc_count()`函数统计点击次数，并使用`sort（）`函数进行排序，使用`random_select`函数对点击次数高于`threshold`的用户进行随机采样10000正用户，文件名为`domain_positive_user.txt`

##### 2.5 将`domain_positive_search_user.txt`整个文件上传至`COSMOS`的`/my`文件夹下,在`ExtractPositive.script`程序中将输出文件名修改`domain_negative_user_time.txt`,将`WHERE`后面的判断条件前加上`NOT`，并添加相应的资源文件。`ExtractPositive.script.cs` 程序中对应的文件名为`domain_positive_search_user.txt`

##### 2.6 将该`domain`下的负向用户下载到本地，使用`random_select`函数对用户进行随机采样10000负用户,文件名为`domain_negative_user.txt`

#### 3 抽取用户搜索记录

##### 3.1 将随机采样好的正负用户样本文件`domain_positive_user.txt`和`domain_negative_user.txt`上传至`COSMOS`的`/my`文件夹下

##### 3.2 修改COSMOS上的`ExtractPosInfo.script`程序中对应输出和资源的文件名分别为`domain_positive_user_info.txt`和`domain_positive_user.txt`，在`ExtractPositive.script.cs` 程序中对应的文件名为`domain_positive_user.txt`，抽取正向用户一个月内所有的数据记录

##### 3.3 修改COSMOS上的`ExtractPosInfo.script`程序中对应输出和资源的文件名分别为`domain_negative_user_info.txt` 和`domain_negative_user.txt`， `ExtractPositive.script.cs` 程序中对应的文件名为`domain_negative_user.txt`，抽取负向用户一个月内所有的数据记录

##### 3.4 将正负向用户搜索记录下载至本地，分别保存为`domain_positive_user_info.txt`和`domain_negative_user_info.txt`.

#### 4 用户搜索记录处理

##### 4.1 在`pre.py`文件中，调用`split_web()`函数，切分训练网站和测试网站，保存文件，分别命名为`domain_website_train.txt`和 `domain_website_test.txt`

##### 4.2 在`pre.py`文件中，调用`split_user()`函数，切分训练用户和测试用户，并求出两边用户的交集，对交集用户做出适当切分，分别保存文件为`domain_positive_user_train.txt`和 `domain_positive_user_test.txt`

##### 4.3 在`pre.py`文件中，调用`split_info()`函数，切分训练用户搜索记录和测试用户搜索记录，保存文件，分别命名为`domain_positive_info_train.txt`和 `domain_positive_info_test.txt`

#### 5 构建训练数据集

###### 5.1 将输入文件设置为`domain_positive_info_train.txt`，调用`pre.py`程序中的`build_data()`函数，通过返回`combine_records`和`new_combine_record`区分仅去除标注网站和去除标注网站与强相关搜索query的记录

###### 5.2 调用`user_aggragation`函数，并将`combine_records`（或`new_combine_record`）作为参数传入，进行用户搜索记录按照用户的聚合

###### 5.3 调用`write_dict`函数完成正向用户训练集的构建

###### 5.4 将输入文件设置为`domain_positive_info_test.txt`，重复5.1-5.3,完成正向用户测试集的构建

###### 5.5 将输入文件设置为`domain_negative_info.txt`，并将5.3中第一个参数由1改为0，重复5.1-5.3,完成负向用户构建

###### 5.6 调用`pre.py`函数中`combine_pn()`函数，分别构建训练集和测试集

#### 至此完成所有Bing数据预处理工作，之后调用HURA.py文件，进行训练测试



### UET数据抽取及预处理

#### 1 网站选取及处理

##### 1.1 选定关于某个domain的官方网站，保存为文件`domain_website.txt`

##### 1.2 将`domain_website.txt`的文件上传至`COSMOS`的`/my`文件夹下

#### 2 抽取用户及处理

##### 2.1 修改COSMOS上的`UETPositive.script`程序中对应输出和资源的文件名分别为`domain_positive_user_time.txt`和`domain_website.txt`,  `UETPositive.script.cs` 程序中对应的文件名为`domain_website.txt`

##### 2.2 运行`UETPositive.script`程序获取该domain下的`positive user`

##### 2.3 将该`domain`下的正向用户下载到本地,并依次调用`extract_id()`函数抽取用户id，`extract_pos_user（）`函数抽取访问次数高于`threshold`的用户，将文件命名为`domain_positive_user_threshold.txt`

##### 2.4 调用`pre.py`文件中的`random_select()`函数,对正向用户进行采样,保存抽取后的正向用户,将文件命名为`domain_p_u_threshold.txt`

##### 2.5 将`domain_positive_user_time.txt`文件上传至`COSMOS`的`/my`文件夹中,在`UETPositive.script`程序中对应输出的文件名为`domain_negative_user_time.txt`, 将`WHERE`后面的判断条件前加上`NOT`，并添加相应的资源`domain_positive_user_time.txt`, 在`UETPositive.script.cs` 程序中对应的文件名为`domain_positive_user_time.txt`.

##### 2.6 运行`UETPositive.script`程序,并将负向用户列表下载到本地进行去重处理

##### 2.7调用`random_select()`函数,并将输入输出文件名中的positive更改为negative,构建负向用户集

#### 3 抽取用户UET记录

##### 3.1 将`domain_p_u_threshold.txt`和`domain_n_u.txt`上传至`COSMOS`的`/my`文件夹下

##### 3.2 将程序`UETPosInfo.script`中的输出文件名设置为`domain_p_info_threshold.txt`,并添加资源文件`domain_p_u_threshold.txt`,在`UETPosInfo.script.cs`程序中,修改文件输入为`domain_p_u_threshold.txt`,抽取正向用户所有UET数据记录

##### 3.3 将程序`UETPosInfo.script`中的输出文件名设置为`domain_n_info.txt`,并添加资源文件`domain_n_u.txt`,在`UETPosInfo.script.cs`程序中,修改文件输入为`domain_n_u.txt`,抽取负向用户所有UET数据记录

##### 3.4 将正负向用户搜索记录下载至本地，分别保存为`domain_p_info_threshold.txt`和`domain_n_info.txt`

#### 4 用户UET记录处理及构建完整数据集

##### 4.1 在`pre.py`文件中，调用`reduce_label_info()`函数，清理标注网站记录，保存文件，并命名为`domain_p_clean_info_threshold.txt`

##### 4.2 在`pre.py`文件中,调用`id_titleMAP()`函数完成id-url-title的映射匹配,输出文件名为`domainIdTitle_threshold.txt`

##### 4.3 在`pre.py`文件中,调用`bulid_pos_data()`函数完成正向用户数据的用户聚合,输出文件名为`domain_pos_train_threshold.txt`

##### 4.4 重复4.2-4.3,将输入文件改为`domain_n_info.txt`,完成负向用户数据的用户聚合,输出文件名为`domain_neg_train.txt`

##### 4.5 调用`pre.py`文件中的combine_pn(),完成构建训练集

#### 至此完成所有UET数据预处理工作，之后调用HURA.py文件，进行训练测试