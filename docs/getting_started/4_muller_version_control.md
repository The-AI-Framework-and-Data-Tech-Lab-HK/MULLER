GTN-F提供与Git相似的指令来管理数据集的变更，它可以作用于任何大小的数据集，并记录数据集演变的关键信息。详细API文档可参考：[[Dataset Version Control](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405173567643))]。
以下是几个关键指令的介绍：

#### 5.1. Commit

每次对数据集做增删改等写入变更操作后，可使用`ds.commit()`用于将本次变更生成版本ID。示例如下。

加载GTN-F数据集，创建tensor列，并提交变更。

```python
>>> ds = gtn_f.dataset("/data/gtn_f_exp/", overwrite=True)
>>> ds.create_tensor(name="labels", htype="generic", dtype="int")
>>> first_commit_id = ds.commit(message="first commit.")
>>> first_commit_id
'firstdbf9474d461a19e9333c2fd19b46115348f'
```

接下来可以继续在原有tensor列上添加/删除/更新数据，并提交变更。

```python
>>> ds.labels.extend([1,2,3,4,5])
>>> second_commit_id = ds.commit(message="add labels.")
>>> second_commit_id
'637adeb5232f0152b866c9a2a49e8da19f00c1da'
```

* `ds.commit()`接口的具体使用方式与示例可参考：[[dataset.commit()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405213594128)]

#### 5.2. Checkout

每个初创建数据集默认分支都是main分支。可使用`ds.checkout()`创建新的分支（branch）或切换到已有分支。示例如下。

创建新分支：设置参数`create=True`。

```python
>>> ds.checkout("dev", create=True) 
>>> ds.create_tensor("categories", htype="text")
>>> ds.categories.extend(['agent', '情感', '生成', '写作', '情感'])
>>> ds.commit("created categories tensor column and add values.")
>>> ds.summary()  # 现在，在dev分支上有两个tensor列（categories和labels）。
tensor     htype    shape    dtype  compression
 -------    -------  -------  -------  ------- 
categories   text    (5, 1)     str     None   
  labels    generic  (5, 1)    int64    None
```

切换到已有分支：

```python
>>> ds.checkout("main") # 切换到之前的main分支。
>>> ds.branch #查看当前dev名称
'main'
>>> ds.summary()  # 这个分支只有一个tensor列（labels）。
tensor    htype    shape    dtype  compression
-------  -------  -------  -------  ------- 
labels   generic  (5, 1)    int64    None
```

* checkout接口的具体使用方式可参考：[[dataset.checkout()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405213594135)]
* branch接口的具体使用方式可参考：[[dataset.branch](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405213594151)]

#### 5.3. 加载特定版本与分支、查看历史提交

您可使用`gtn_f.load()`方法加载数据集对应的分支和版本。

默认加载数据集main分支版本：

```python
ds = gtn_f.load(path="/data/gtn_f_exp/")
```

亦可加载数据集的特定分支（如有），使用@{branch_name}即可：

```python
ds = gtn_f.load(path="/data/gtn_f_exp@dev")
```

亦可加载数据集的特定commit版本（如有），使用@{commit_id}即可：

```python
ds = gtn_f.load(path="/data/gtn_f_exp@3e49cded62b6b335c74ff07e97f8451a37aca7b2")
```

`log()`方法将日志**打印**到控制台，并返回从当前版本开始的提交记录。

```python
>>> ds = gtn_f.load(path="/data/gtn_f_exp@dev")
>>> ds.log()---------------
GTN_F Version Log
---------------

Current Branch: dev
Commit : 476be1766915dfe652f39e39f5370f9c04528df9 (dev) 
Author : public
Time   : 2025-02-28 08:07:48
Message: created categories tensor column and add values.

Commit : 637adeb5232f0152b866c9a2a49e8da19f00c1da (main) 
Author : public
Time   : 2025-02-28 08:03:16
Message: add labels.

Commit : firstdbf9474d461a19e9333c2fd19b46115348f (main) 
Author : public
Time   : 2025-02-28 03:32:53
Message: fisrt commit.
```

* `log()`方法的具体使用方式可参考：[[dataset.log()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624317)]
  【注：如有需要，我们也会更新`log()`方法，使其也返回*被merge到当前分支的其他分支的历史提交信息*】
* 其他可查看版本管理历史记录信息的方法包括：

> * 获取某个commit的具体信息[[daraset.get\_commit\_details(commit_id)](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624325)]
> * 获取当前commit_id[[daraset.commit_id](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624339)]
> * 获取下一次（还未提交的）commit_id[[daraset.pending_commit_id](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624348)]
> * 判断当前是否有未提交（未commit）的增删改变化[[daraset.has\_head\_changes](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624352)]
> * 以列表形式返回所有commit提交记录[[daraset.commits](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624355)]【与`log()`方法相似，但不打印到控制台】
> * 以列表形式返回所有分支名称[[daraset.branches](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624359)]
> * 基于给定的版本号或者分支名称，返回二者之间的版本提交记录 [[dataset_commits_between()](...)]

#### 5.4. Direct Diff【在v0.6.10以上版本提供】

了解版本之间的变换对管理数据变更是至关重要的，GTN-F提供 `ds.direct_diff(id_1, id_2)` 接口来获取<font color="red">`id_2`版本中对应`id_1`版本的</font>每列每行数据的直接差异，可将这个差异返回为pandas dataframe对象以便于用户查看。

`ds.direct_diff()` 接口接口详细参数如下。注意：<font color="red">`id_1`和`id_2`有先后顺序之分！</font>

* **id_1 (str)** - 待比较的第一个版本号或者分支名称。
* **id_2 (str)** - 待比较的第二个版本号或者分支名称。
* **as_dataframe (bool, Optional)** - 当声明为True的时候可返回dataframe对象
* **force (bool, Optional)** - 多于100000行（TO\_DATAFRAME\_SAFE\_LIMIT的值，可以在contants里修改）的dataframe生成时间可能较长（大于5s），需用户手动设置force=True来确认执行。

使用示例如下。

```python
import gtn_f
ds = gtn_f.dataset(path="temp_test", overwrite=True)

ds.create_tensor(name="labels", htype="generic", dtype="int")
ds.create_tensor(name="categories", htype="text")
ds.create_tensor(name="test1", htype="generic", dtype="int")
ds.labels.extend([0, 1, 2, 3, 4])
ds.categories.extend(["a", "b", "c", "d", "e"])
ds.test1.extend([100, 101, 102, 103, 104])

ds.checkout("dev", create=True)
ds.labels[0] = 11
ds.categories[3] = "haha"
ds.pop(2)
ds.labels.append(100)
ds.categories.append("hello")
ds.test1.append(105)
dev_1 = ds.commit("first on dev")

ds.checkout("main", create=False)
ds.checkout("dev_2", create=True)
ds.delete_tensor("test1")
ds.labels.extend([5, 6])
ds.categories.extend(["f", "g"])
ds.commit("first on dev_2")
ds.labels[1] = 111
ds.categories[1] = "xixixi"
ds.create_tensor(name="test2", htype="generic", dtype="int")
ds.test2.extend([100, 101, 102, 103, 104, 105, 106])
dev_2 = ds.commit("sec on dev_2")

final_df_dict = ds.direct_diff(dev_1, dev_2, as_dataframe=True)
```

在jupyter notebook环境下可直观看出dev_1分支与dev_2分支的变化。
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202502286098802/20941013/07d51274598d4de7b59f320195777fb1.png)

#### 5.5. Diff

了解版本之间的变换对管理数据变更是至关重要的，GTN-F提供 `diff()` 接口来获取两个版本中每一个tensor列中样本（即每列每行）的差异。

* 注1：如果用户执行`diff()`时所在的状态是待比较的版本之一，并且当前状态是当前分支的HEAD_NODE，那么当前状态未提交（commit）的变更也会参与diff计算。
* 注2：此`diff()` 接口主要是为`merge()`接口的计算服务。因此，此接口获取的<font color="red">并不是两个版本之间的绝对差异，而是两个版本距离他们的**最近祖先之间的每个版本提交的差异**（与log类似），是以版本树的形式返回的</font>！
* 注3：`diff()` 接口默认只返回数据变动的<font color="red">行号</font>，而非真实值。如需要返回变动的真实值（比如labels列的第3行“从10更新为20”），则需要使用`show_value=True`参数，具体使用方式可参考下文。注意加载真实值涉及io操作，如数据存储在带宽较小的云端（如华山公有云），则<font color="red">建议配合`offset`与`limit`参数共同使用</font>，避免一次加载过多真实值而造成效率低下。【<font color="red">注意：因需要加载真实值，加载全部diff的话需要较大内存，建议（1）如果用6G以下的内存的话，善用offset和limit参数，不要一次加载全部diff的真实数据；或（2）如在大数据量情况下必须加载全部diff的真实数据，可考虑给多一点内存资源，如24G以上，避免OOM</font>】
* 注4：使用参数`as_dict=True`可以以字典类型返回这些差异。

`diff()` 有三种使用方式：

使用方式1：获取当前状态和上一个（已提交）版本之间的差异

```python
ds.diff() #注：如果在上一个版本commit（提交）之后您没有任何操作，那么这里将不会返回变更内容。
```

使用方式2：获取当前状态和 id_1 版本之前的差异

```python
ds.diff(id_1=<commit_id>) # id_1可以是某个commit_id
ds.diff(id_1=<branch name>) # id_1也可以是某个分支的名字，代表这个分支上的最新commit
```

使用方式3：获取id_1 和 id_2 之间的差异

```python
ds.diff(id_1=<commit_id_1>, id_2=<commit_id_2>)  # id_1和id_2可以是commit_id
ds.diff(id_1=<branch name_1>, id_2=<branch name_2>)  # id_1和id_2也可以是分支名，分别代表这些分支上的最新commit
```

* 注1：`diff()`参数选择较多：

> * id\_1 (str, Optional): 待比较的第一个版本号或者分支名称。
> * id\_2 (str, Optional): 待比较的第二个版本号或者分支名称。
> * as\_dict (bool, Optional): 如果设置为True，则用list的形式返回版本间的变动。默认为False。
> * show\_value (bool): 如果设置为`True`，将返回增加（append），更新（update）和删减(pop)的变动的<font color="red">真实数据</font>。
> * offset(int): 需要跳过的展示数据数量，仅在将show\_value设置为True的时候发挥作用。默认为0。
> * limit(int): 展示数据的总数目，仅在将show\_value设置为True的时候发挥作用。默认为1000。
> * asrow (bool): 以行的形式返回用于展示的数据，仅在将show\_value设置为True的时候发挥作用。默认为False。如果设置为True，将以行的形式返回样本，返回的格式为列表（list），列表内的字典（dict）数量即为行的数量。如果设置为False，将以列的形式返回数据，返回的格式为字典（dict），字典内根据张量将数据存在列表（list）中返回。【注：本参数只在两个版本的<font color="red">**tensor列数量和行数量的变动完全相同**</font>的时候适用，否则会抛changes in tensors are different的错误！故更建议用户将该参数改为False，<font color="red">**在单列更新等情况下使用其他参数灵活处理**</font>。】

* 注2：

> 1. 如果id\_1=None, id\_2=None:
>    将默认返回当前branch的HEAD节点与主分支公共祖先之间的提交记录
>    若当前branch为主分支，将抛出valueerror报错
> 2. 如果只有id\_1=None
>    将默认返回id\_2（若id\_2为分支名，则使用该分支的最新节点）与主分支公共祖先之间的提交记录
>    若id\_2处于主分支，将抛出valueerror报错
> 3. 如果只有id\_2=None
>    将默认返回id\_1（若id\_1为分支名，则使用该分支的最新节点）与主分支公共祖先之间的提交记录
>    若id\_1处于主分支，将抛出valueerror报错
> 4. 如果id\_1 与 id\_2 都不为None
>    将返回id\_1与id\_2之间的版本提交记录（若为分支名，则使用该分支的最新节点）

以5.2章节中的创建的数据集为例：

默认不展示diff的具体值

```python
>>> ds.diff("dev", "main")  # 默认不展示值
## GTN_F Diff
The 2 diffs are calculated relative to the most recent common ancestor (637adeb5232f0152b866c9a2a49e8da19f00c1da) of the two commits passed.
------------------------------------------------------------------------------------------------------------------------
Diff in dev (target id 1):

********************************************************************************
commit UNCOMMITTED HEAD
Author: public
Date: None
Message: None

No changes were made in this commit.
********************************************************************************
commit 476be1766915dfe652f39e39f5370f9c04528df9
Author: public
Date: 2025-02-28 08:07:48
Message: created categories tensor column and add values.

categories
* Created tensor
* Added 5 samples: [0-5]

------------------------------------------------------------------------------------------------------------------------
Diff in main (target id 2):

********************************************************************************
commit UNCOMMITTED HEAD
Author: public
Date: None
Message: None

No changes were made in this commit.
------------------------------------------------------------------------------------------------------------------------
{}
>>>
```

展示diff的具体值

```python
>>> ds.diff(id_1=third_commit_id, id_2=first_commit_id, show_value=True)
## GTN_F Diff
The 2 diffs are calculated relative to the most recent common ancestor (firstdbf9474d461a19e9333c2fd19b46115348f) of the two commits passed.
------------------------------------------------------------------------------------------------------------------------
Diff in 51f5bb41c4a1d74d881269366e3718285b8145db (target id 1):

********************************************************************************
commit 51f5bb41c4a1d74d881269366e3718285b8145db
Author: public
Date: 2025-03-03 04:03:13
Message: created categories tensor column and add values.

categories
* Created tensor
* Added 5 samples: [0-5],
The appended data values are {'created': True, 'cleared': False, 'info_updated': False, 'data_added': [0, 5], 'data_updated': set(), 'data_deleted': SortedSet([]), 'data_deleted_ids': [], 'data_transformed_in_place': False, 'add_value': [array(['agent'], dtype='<U5'), array(['情感'], dtype='<U2'), array(['生成'], dtype='<U2'), array(['写作'], dtype='<U2'), array(['情感'], dtype='<U2')], 'updated_values': [], 'data_deleted_values': []}

********************************************************************************
commit 6536edb5ee077de334d6f23133bcf7e209d1be80
Author: public
Date: 2025-03-03 04:03:12
Message: add labels.

labels
* Added 5 samples: [0-5],
The appended data values are {'created': False, 'cleared': False, 'info_updated': False, 'data_added': [0, 5], 'data_updated': set(), 'data_deleted': SortedSet([]), 'data_deleted_ids': [], 'data_transformed_in_place': False, 'add_value': [array([1]), array([2]), array([3]), array([4]), array([5])], 'updated_values': [], 'data_deleted_values': []}

------------------------------------------------------------------------------------------------------------------------
Diff in firstdbf9474d461a19e9333c2fd19b46115348f (target id 2):

No changes were made.

------------------------------------------------------------------------------------------------------------------------
{}
```

以dict形式返回diff的真实值

```python
>>> ds.diff(id_1=third_commit_id, id_2=first_commit_id, as_dict=True, show_value=True)
{'dataset': ([{'commit_id': '51f5bb41c4a1d74d881269366e3718285b8145db',
    'author': 'public',
    'message': 'created categories tensor column and add values.',
    'date': '2025-03-03 04:03:13',
    'info_updated': False,
    'renamed': OrderedDict(),
    'deleted': [],
    'commit_diff_exist': True,
    'tensor_info_updated': False},
   {'commit_id': '6536edb5ee077de334d6f23133bcf7e209d1be80',
    'author': 'public',
    'message': 'add labels.',
    'date': '2025-03-03 04:03:12',
    'info_updated': False,
    'renamed': OrderedDict(),
    'deleted': [],
    'commit_diff_exist': True,
    'tensor_info_updated': False}],
  []),
 'tensor': ([{'commit_id': '51f5bb41c4a1d74d881269366e3718285b8145db',
    'categories': {'created': True,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 5],
     'data_updated': set(),
     'data_deleted': SortedSet([]),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [array(['agent'], dtype='<U5'), array(['情感'], dtype='<U2'), array(['生成'], dtype='<U2'), array(['写作'], dtype='<U2'), array(['情感'], dtype='<U2')],
     'updated_values': [],
     'data_deleted_values': []},
    'labels': {'created': False,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 0],
     'data_updated': set(),
     'data_deleted': set(),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [],
     'updated_values': [],
     'data_deleted_values': []}},
   {'commit_id': '6536edb5ee077de334d6f23133bcf7e209d1be80',
    'labels': {'created': False,
     'cleared': False,
     'info_updated': False,
     'data_added': [0, 5],
     'data_updated': set(),
     'data_deleted': SortedSet([]),
     'data_deleted_ids': [],
     'data_transformed_in_place': False,
     'add_value': [array([1]), array([2]), array([3]), array([4]), array([5])],
     'updated_values': [],
     'data_deleted_values': []}}],
  [])}
```

* `diff()`的具体使用方式可参考：[[dataset.diff()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405213594134)] [[详细MR](https://codehub-y.huawei.com/GTN/GTN-F/merge_requests/157)]

#### 5.6. 什么是HEAD变更？如何使用reset撤销未提交的变更？

与Git不同，GTN-F的版本控制<font color="red">没有本地暂存，所有的变更都会立刻同步到永久存储的位置（包括本地和云端）</font>。因此，数据集的任何变更都会自动更新到当前分支的HEAD node。这意味着，虽然未提交的变更不会出现在其他分支，但所有在当前分支的用户都能永久访问未提交的变更。

在下例中，最后我们回到main分支，有一个 'lables' tensor列，且有5个样本。此时再次添加1个样本，并通过 `has_head_changes`属性判断当前状态 HEAD 有变更。

```python
>>> ds = gtn_f.dataset(path="temp_dataset/",overwrite=True)
>>> with ds:
        ds.create_tensor("labels", htype="generic")
        ds.labels.extend([1, 2, 3, 4, 5])
>>> ds.commit()
>>> ds.checkout("dev", create=True)
>>> ds.checkout("main")
>>> with ds:
        ds.labels.append(100)
>>> ds.has_head_changes
True
```

当切换至dev分支时，'labels' tensor列 只有5个样本。

```python
>>> ds.checkout('dev')
>>> print('Dataset in {} branch has {} samples in the labels tensor'.format(ds.branch, len(ds.labels)))
Dataset in dev branch has 5 samples in the labels tensor
```

当我们再切换至main分支时，未提交的变更依然存在，此时'labels' tensor列 有6个样本。

```python
>>> ds.checkout('main')
>>> print('Dataset in {} branch has {} samples in the labels tensor'.format(ds.branch, len(ds.labels)))
Dataset in main branch has 6 samples in the labels tensor
>>> ds.has_head_changes
True
```

使用 `reset()` 方法可以撤销在当前分支下未提交的变更。执行后，'labels' tensor列 恢复到5个样本。

```python
>>> ds.reset()
>>> ds.has_head_changes
False
>>> print('Dataset in {} branch has {} samples in the labels tensor'.format(ds.branch, len(ds.labels)))
Dataset in main branch has 5 samples in the labels tensor
```

* `reset()`的具体使用方式可参考：[[dataset.reset()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624324)]

#### 5.7. Merge【在v0.6.7以上版本提供】

合并分支是进行协同开发的关键操作。与Git相似，GTN-F的分支合并方式可以分为两种：

* **Fast-forwad Merge 前向合并（无冲突）**
  如下图，我们基于main分支创建（checkout，create=True）新的dev分支后，<font color="red">main分支无任何变化</font>。因此我们稍后在main分支合入dev分支时，dev分支具有main分支的所有提交行为（即领先于main分支），<font color="red">无需解决冲突</font>。因此不需要创建新的提交或修改main分支，即可顺利合入。
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202502286098802/18817110/a6368739634942009e77a40b2a5073a3.png)
* **3-way Merge 三路合并（可能有冲突）**
  如下图，我们基于main分支创建（checkout，create=True）新的dev分支后，<font color="red">main分支发生了变化</font>。因此我们稍后在main分支合入dev分支时，由于main分支具有dev分支没有的提交行为，所以这两个分支<font color="red">有可能存在冲突</font>。此时需要（自动）创建一个新的提交，该提交将两个分支合并在一起。
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202502286098802/18817993/46f6080748ef40e484f5073a2300bf8b.png)
  这种三路合并的情况，在多用户多分支开发的标准开发流程中比较常见。如下图的情况，基于main分支创建了新的dev_1与dev_2分支后，两个子分支各自进行开发迭代。最终main分支先合入dev_1分支，之后main分支即产生了变动，也就可能与dev_2分支产生冲突。换句话来说，在`ds.merge("dev_1")`命令之后，就是一种可能存在冲突的三路合并。
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202502286098802/18817184/d8f11d85986e4bd0adc70a870e5e89a7.png)
  
  GTN-F现遵循以下合入流程：
  
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202502286098802/18820904/0423733ad6ac44dfb212878af161fcd7.png)

1. <font color="red">**用户通过`ds.detect_merge_conflict(target_id='xx')`接口查看可能产生的冲突**</font>，大致原理如下：
   （1）找到当前分支HEAD node M3版本与待合入分支最新版本D4版本的共同祖先——M1版本。
   （2）获取D4版本与M1版本的差异`diff_1`，获取M3版本与M1版本的差异`diff_2`【这里涉及同一分支上的多个版本差异计算，细节在此省略】。
   （3） 对于D4版本与M3版本共同的tensor列，我们比较每一个tensor列下的每个样本id（sample global id）及其唯一标识（uuid），<font color="red">基于每行数据的uuid查看是否存在冲突</font>。如完全没有冲突，这一步将返回None。如存在冲突，这一步将打印并在内存lru_cache中记录可能存在的冲突：
   
   * pop冲突：如果D4与M3中<font color="red">都执行了pop操作，且pop的是不同的行</font>，则这里会记录（1）D4（相比于M1）执行了哪几行的pop，（2）M3（相比于M1）执行了哪几行的pop。均返回这些行在D4或M3版本内的global sample id和对应的值。【注意：如果两者都执行了相同行的pop，则不会产生冲突】
   * update：如果D4与M3中<font color="red">都执行了相同行的update操作</font>，则这里会记录：（1）D4执行了哪几行的update，及其更新的值。（2）M3执行了哪几行的update，及其更新的值。均返回这些行在D4或M3版本内的global sample id和对应的值。【注意：如果两者是执行不同行的update操作，则不会产生冲突】
   * append冲突：如果D4与M3中<font color="red">都执行了append操作</font>，则这里会记录（1）D4 append了多少行及其值，（2）M3 append了多少行及其值。均返回这些行在D4或M3版本内的global sample id和对应的值。
2. <font color="red">**用户通过`ds.merge(target_id='xx')`合并分支**</font>，注意这里有三类参数可供用户决定合并策略。
   （1）解决冲突相关的参数：
   
   * `append_resolution` 参数, 有以下四种选择。
     > * None：默认值，当存在append冲突时直接抛出异常。
     > * "ours": 采用当前分支的append操作。
     > * "theirs": 采用目标分支的append操作。
     > * "both": 当前分支和目标分支的append操作都采用。
   * `update_resolution` 参数，有以下三种选择。
     > * None：默认值，当存在update冲突时直接抛出异常。
     > * "ours": 采用当前分支的update操作。
     > * "theirs": 采用目标分支的update操作。
   * `pop_resolution` 参数, 有以下四种选择。
     > * None：默认值，当存在pop冲突时直接抛出异常。
     > * "ours": 采用当前分支的pop操作。
     > * "theirs": 采用目标分支的pop操作。
     > * "both": 当前分支和目标分支的pop操作都采用。

注1：在数据集维度上，pop和append往往更改了数据集的长度和global index的值（用pop举例，pop一行样本之后，所有后面的样本的global sample index都减一，因此不同版本之间的变化难以对应）。因此我们是<font color="red">基于uuid来做冲突的检测和解决</font>。所以建议大家灵活运用ds.pre_merge()接口和版本数据增删改功能，确认过改动无冲突之后，再使用ds.merge()接口进行合并。
注2：在冲突解决的流程中，以上三种冲突解决的<font color="red">执行顺序会影响最后数据集的结果</font>。

（2）`delete_removed_tensors`参数，默认为False。

* 如设置为True，则默认舍弃`diff_2`中删除的tensor列。

（3）`force`参数，默认为False。 举例来说`force = True`在以下情况会产生这些影响：

* 如果在D4版本上重命名了tensor列，但M3版本中缺少tensor列，则重命名后的tensor列将在当前分支上注册为新tensor列。
* 如果tensor列在D4版本和M3版本中都被重命名，则D4版本上的tensor列将被注册为当前分支上的新tensor列。
* 如果在D4版本上重命名tensor列，并且在当前分支上创建了新名词的新tensor列，则它们将被合并。

<font color="red">注意，用户需要仔细区分当前是否有冲突。</font>举例来说，假设用户在main分支上有20条数据，然后checkout一个新分支dev。如果用户在dev分支上删除了第2条数据，并checkout回到main分支并merge dev分支，则此时会有冲突。因为main分支上的第2条数据还是存在的。用户可考虑使用参数`pop_resolution="theirs"`来采用dev分支上的变更。

以下为使用示例：

```python
# 创建数据集，并在main分支添加数据
>>> ds = gtn_f.dataset(path="temp_test", overwrite=True)
>>> ds.create_tensor(name="labels", htype="generic", dtype="int")
>>> ds.labels.extend([0, 1, 2, 3, 4])
>>> ds.create_tensor(name="categories", htype="text")
>>> ds.categories.extend(["a", "b", "c", "d", "e"])
# 创建dev-1分支，并进行添加、修改、删除操作。
>>> ds.checkout("dev-1", create=True)
>>> ds.labels.extend([50,60,70])
>>> ds.categories.extend(["ff", "gg", "hh"])
>>> ds.labels[3] = 30
>>> ds.pop(1)
>>> print(ds.labels.numpy())
[[ 0]
 [ 2]
 [30]
 [ 4]
 [50]
 [60]
 [70]]
>>> ds.commit()
# 返回main分支，创建dev-2分支，并进行添加、修改、删除操作。
>>> ds.checkout("main")
>>> ds.checkout("dev-2", create=True)
>>> ds.labels.extend([500, 600, 700, 800])
>>> ds.categories.extend(["fff", "ggg", "hhh", "iii"])
>>> ds.labels[3] = 300
>>> ds.labels[4] = 400
>>> ds.pop([1, 2])
>>> print(ds.labels.numpy())
[[  0]
 [300]
 [400]
 [500]
 [600]
 [700]
 [800]]
>>> ds.commit()
# 返回到main分支，合入dev分支，注：使用dev-1分支的pop
>>> ds.checkout("main")
>>> ds.merge("dev-1", pop_resolution="theirs")
>>> print(ds.labels.numpy())
[[ 0]
 [ 2]
 [30]
 [ 4]
 [50]
 [60]
 [70]]
# 查看此时main分支与dev-2分支的冲突
>>> conflict_tensors, conflict_records = ds.detect_merge_conflict("dev-2", show_value=True)
>>> pprint(conflict_records)
{'categories': {'app_ori_idx': [4, 5, 6],
                'app_ori_values': [array(['ff'], dtype='<U2'),
                                   array(['gg'], dtype='<U2'),
                                   array(['hh'], dtype='<U2')],
                'app_tar_idx': [3, 4, 5, 6],
                'app_tar_values': [array(['fff'], dtype='<U3'),
                                   array(['ggg'], dtype='<U3'),
                                   array(['hhh'], dtype='<U3'),
                                   array(['iii'], dtype='<U3')],
                'del_ori_idx': [],
                'del_ori_values': [],
                'del_tar_idx': [2],
                'del_tar_values': [array(['c'], dtype='<U1')],
                'update_values': {'update_ori': [], 'update_tar': []}},
 'labels': {'app_ori_idx': [4, 5, 6],
            'app_ori_values': [array([50]), array([60]), array([70])],
            'app_tar_idx': [3, 4, 5, 6],
            'app_tar_values': [array([500]),
                               array([600]),
                               array([700]),
                               array([800])],
            'del_ori_idx': [],
            'del_ori_values': [],
            'del_tar_idx': [2],
            'del_tar_values': [array([2])],
            'update_values': {'update_ori': [{2: array([30])}],
                              'update_tar': [{1: array([300])}]}}}
# 合入dev-2分支，可根据不同的解决冲突的参数设置来采用不同的合并策略。
>>> ds.merge("dev-2", append_resolution="both", pop_resolution="ours", update_resolution="theirs")
>>> print(ds.labels.numpy())
[[  0]
 [  2]
 [300]
 [400]
 [ 50]
 [ 60]
 [ 70]
 [500]
 [600]
 [700]
 [800]]
```

* detect_merge_conflict接口的具体使用方式可参考：[[dataset.detect_merge_conflict()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202503176267980)]
* merge接口的具体使用方式可参考：[[dataset.merge()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405213594142)]

#### 5.8. 针对华山平台使用而开发的权限需求功能【在v0.6.6以上版本提供】

针对华山平台场景的权限管控需求，我们新增如下限制：

1. 只有创建主分支的用户对**所有分支**有写入和删除权限。
2. 只有创建主分支的用户对数据集有删除和重命名权限，即以下接口：
   
   ```python
   ds.delete()
   ds.rename()
   ```
3. 非创建主分支的用户只在**自己创建的分支**有写入和删除权限，对其他分支只有只读权限。
   目前涉及上述写入和删除权限（只能在自己的分支操作）的接口有：
   
   ```python
   ds.reset()
   ds.create_tensor()
   ds.create_tensor_like()
   ds.commit()
   ds.extend([dataset对象])
   ds.[tensor].extend()
   ds.delete_branch() 
   ds.rechunk()
   ds.append([dataset对象])
   ds.[tensor].append
   ds.update() #注：有两种update方法
   ds.merge()
   ds.delete_tensor()
   ds.pop()
   ds.rename()
   gtn_f.api_dataset.create_dataset_from_dataframes()
   gtn_f.api_dataset.create_dataset_from_file()
   ds.[tensor].clear()
   ds.create_index()
   ```
4. 用户只能删除自己创建的搜索视图（view），即以下接口：
   
   ```python
   gtn_f.delete_view()
   ```

* 注：实际实现中，非创建主分支的用户对主分支的一个文件（`version_control_info.json`，用于记录版本信息）有写入权限。
