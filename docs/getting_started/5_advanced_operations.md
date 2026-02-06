## Advanced Operations

#### 7.1 新增大数据量到GTN-F数据集的建议

如需要创建或新增大数据量到GTN-F格式的数据集，建议使用`@gtn_f.compute`修饰器并行操作（一般`num_workers`可设为8-32，取决于系统可用资源）。

> * 注意：多进程/多线程启动前会有overhead，在几十万到百万大数据量效果下才有显著效果。
> * 使用示例：

```python
def create_cifar10_dataset_parallel(num_workers=4, scheduler="threaded"):
    ds_multi = gtn_f.dataset(path="./temp_test", overwrite=True)
    with ds_multi:
        ds_multi.create_tensor("test1", htype="text")
        ds_multi.create_tensor("test2", htype="text")

    # 建议以行为单位添加数据，以保证行的原子性
    iter_dict = []
    for i in range(0, 100000):
        iter_dict.append((i, ("hi", "hello")))  # 只是举例，您可在实际操作时读入任意数据

    @gtn_f.compute
    def file_to_gtnf(data_pair, sample_out):
        sample_out.test1.append(data_pair[1][1])
        sample_out.test2.append(data_pair[1][0])
        return sample_out

    with ds_multi:
        file_to_gtnf().eval(iter_dict, ds_multi, num_workers=num_workers, scheduler=scheduler, disable_rechunk=True)

    return ds_multi


if __name__ == '__main__':
    ds = create_cifar10_dataset_parallel(num_workers=4, scheduler="processed")
```

大数据量下，上述例子中的`eval()`方法也可支持`checkpoint_interval=<commit_every_N_samples>`参数，在每`N`个samples之后做一次落盘保存，以防止中途出现宕机错误而需要全部重新处理。原因是落盘时先写data再写meta数据。如果data写一半挂了或还没来得及写完meta就挂了，还能从上一批数据开始（无需从第一个数据开始）。

> * 注：在这种情况下，数据版本保存在`/versions`文件夹下。

注意： 大数据量下不一定每个sample path都是对的，比如有时数据集里某个sample误用了无效路径，或把png图片当jpeg图片上传了（实际上png比jpeg多一个通道，处理方法不同）。你可以选择在这些情况下使用`.eval(... ignore_errors=True)`忽略错误，否则频繁的报错处理会拖累整体数据上传进度。

* `@gtn_f.compute`修饰器的具体使用方式可参考：[[compute](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405283649169)]。
* `eval()`方法的具体使用方式可参考：[[eval()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405283649174)]。

```
FAQ:
Q: 你们有没有考虑更多的容错方面的修改？
Ans：我们有考虑过这种实现：在更新多个文件的时候，可以采取两阶段提交的方式。首先将数据写入到多个tmp文件中，之后对需要修改的文件做备份，然后将tmp文件重新命名为目标文件名，最后删除备份文件。如果中间出了任何exception，就回滚为备份文件的情况。我们用过一个demo做测试，这种方法能避免“add data的时候进程挂了”的问题，不过添加数据的开销(1w条华山的图片为例)时间从90s上升到110s。如果用户对数据的稳定性有需求的话，可以考虑用类似的两阶段提交的方式实现文件写。
```

#### 7.2 请使用`with`语法来提高数据的写入性能！

1. 对GTN-F数据集的所有独立更新都会被立即（经LRUCache，详见`_set_item()`和`flush()`）推送到目标长期存储的位置。当独立更新较多且数据存储在云端时，写入时间会显著增加。比如说下面这种用法，每次调用`.append()`命令的时候都会把更新推送到长期存储。

```python
for i in range(10):
     ds.my_tensor.append(i)
```

2. `with` 语法会显著提高数据写入的性能。这种写法是当`with`内整段代码执行完或本地缓存满了之后才执行推送到长期存储的操作，所以大量减少了分散的写操作。

```python
with ds:
      for i in range(10):
             ds.my_tensor.append(i)  #或其他涉及写的操作，如create,update等等
```

#### 7.3 为什么会出现数据集损坏的情况？如何处理？

当运行代码意外被中断时（比如新增数据、删除数据时宕机了或被强行中断），数据集会损坏，因为这时有可能出现某些列已经成功执行append/pop、但其他列未成功执行的情况。这时我们可以灵活运用`ds.reset()`接口，撤回未commit的非法操作，回到最近一个数据集合法版本。

1. **场景A：**GTN-F数据集或某些张量列可能出现无法被读取的情况（例如出现下面的警告）。

````
DatasetCorruptError: Exception occured (see Traceback). The dataset maybe corrupted. Try using `reset=True` to reset HEAD changes and load the previous commit. This will delete all uncommitted changes on the branch you are trying to load.
````

* 当出现这种情况，可以用`reset=True`重新加载数据集。

```python
ds = gtn_f.load(<dataset_path>, reset=True)
```

2. **场景B：**GTN-F数据集可能出现损坏（例如出现下面的警告：列长度不一致）。
   ![](https://codehub-y.huawei.com/api/codehub/v1/projects/3032628/uploads/4b5f77b7-bf71-4fa1-a7d7-8a08e0f99944/1726193448394.png)

* 当出现这种情况，可以用`check_integrity=False`与`ds.reset()`重新加载数据集。

```python
ds = gtn_f.load(<dataset_path>, check_integrity=False)  # 不检查数据集的完整性
ds.reset()
```

* 数据集加载时`check_integrity`设置具体文档可参考[[gtn_f.dataset()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405133525071)]与[[gtn_f.load()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405133525129)]。
* reset接口的具体使用方式可参考[[dataset.reset()](https://wiki.huawei.com/domains/72007/wiki/113528/WIKI202405243624324)]。
* 注：一旦reset了，所有uncommitted changes都会被自动删除。
* 显然，在大数据量的情况下，比较好的做法是<font color="red">**使用checkpoint或经常commit**</font>。这样比较方便我们在数据集意外损坏的情况下撤销不合法的更改。

#### 7.4 在OBS侧高效操作的关键

1. 有足够丰富的obs_client接口和足够大的带宽。

* GTN-F最友好的使用方式是 直接“本地->本地”或 “本地->OBS桶”上操作。如果用户在华山现网环境中采用“huashan：//”前缀或无前缀的路径，则是采用这样的调用链“本地 -> OBS桶A（个人桶）-> OBS桶B（共享桶）”，其中桶之间的基于有限的obs接口的频繁读写会严重影响处理效率。
* 我们也期望华山可提供更多丰富的底层OBS操作接口（如批量文件读写、批量文件删除、文件按offset读写等），以提高小文件传输的效率。期望提供的接口列表可参考[[boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)] [[Huawei OBS](https://github.com/huaweicloud/huaweicloud-sdk-python-obs/tree/master/examples)]。

> * 注意： OBS侧上大量小文件传输是个问题。可考虑提供partial read/write功能(如[[Huawei OBS的partial read功能(提供offset)](https://github.com/huaweicloud/huaweicloud-sdk-python-obs/blob/master/examples/concurrent_upload_part_sample.py)])。

2. 有足够大的内存，这样我们的LRUCache可存储的数据量会多一些，一次性flush的数据量也多。如有需要，可通过修改constants.py内`DEFAULT_MEMORY_CACHE_SIZE`的值来增加默认内存值（默认为20GB）。

> * 注：其中1是2的前提。

#### 7.5 并发处理 - GTN-F中的写锁

GTN-F库本身已可<font color="red">**基于文件锁**</font>提供基础的并发写处理，防止多人写冲突。以下三种锁适用于所有使用场景（包括华山notebook环境）。

1. `version_control_info.lock`文件：由于各分支用户都拥有对`version_control_info.json`的写入，故需要这个文件锁来保证一次只有一位用户在写入，其他用户需要等待写入完成、锁释放之后才可进行写操作。
2. `dataset_lock.lock`文件：一旦有用户在某个路径内创建了GTN-F数据集，则该路径会自动生成`dataset_lock.lock`文件锁。在这之后，只要这个文件锁仍存在，则不能再在这个路径内再使用ds.empty()等接口再创建数据集，否则会报以下DatasetHandleError：

```
gtn_f.util.exceptions.DatasetHandlerError: A dataset already exists at the given path (temp_dataset/). If you want to create a new empty dataset, either specify another path or use overwrite=True. If you want to load the dataset that exists at this path, use gtn_f.load() instead.
```

3. `queries.lock`文件：有两个使用场景。（1）在ds.save_view()开始执行时触发，生成该文件锁；等ds.save_view()执行完成时释放锁；从而避免这个view保存的过程中的就有人要使用。（2）ds.delete_view()开始执行时被触发，生成该文件锁；等ds.delete_view()执行完成时释放锁；从而避免这个view被删除的时候还有人在使用。不过现阶段只有生成view的用户可以删除这个view，所以这里提到的极端场景其实可以避免。

但是文件锁的做法还是存在一定的缺点：
（1）文件锁的生成与删除效率取决于华山obs（nsp）的文件读写效率。
（2）文件锁操作的原子性取决于华山obs（nsp）读写接口的原子性。

GTN-F在华山wisefunction端（即创建及加载数据集时）已经支持<font color="red">**基于redis的分布式锁**</font>。现在共支持三种锁：

1. 分支头锁：用于锁某分支的头节点资源。
2. 版本控制锁：用于锁整个数据集的不同版本的公共资源——版本控制信息。
3. 分支锁：用于锁整个分支的前序版本（其实在v0.6.7及以下版本暂时不需要这个）
   为了避免发生死锁，如果外部有需要同时获取多个类型的锁的情况，请严格按照1/2/3的顺序依次获得。

注意：现阶段GTN-F没有读锁，所以我们<font color="red">**允许脏读**</font>（即用户A在对数据集做修改的过程中，若同一时刻用户B读用户A分支，有可能读到不全的数据-即当下的数据集修改的状态）。

#### 7.6 GTN-F与Deeplake有什么不同之处？

Deeplake是个闭源项目。GTN-F格式沿用Deeplake呈现的部分接口，采用自己设计的文件组织形式，对版本管理、加载、OBS支持方面有大量性能优化与重构。
另外较大的不同之处在于：

1. 文件组织形式（与Deeplake的文件组织形式不同，GTN-F自研发，读写效率较高）
2. 基于向量化加速的数据检索（Deeplake无此功能，GTN-F完全自研发）
3. 版本管理功能（采用相似的git for data原理，GTN-F的merge与diff属于新增的自研发功能，支持更复杂的版本管理操作）
4. 多用户并发操作处理与并发锁（Deeplake无此功能，GTN-F完全自研发）
5. 用户分支权限管控（Deeplake无此功能，GTN-F完全自研发）
6. High performance Dataloader（GTN-F完全自研发）
7. 适配华山基于NSP接口的OBS文件操作，文件可直接保存于华山公有云OBS桶并进行操作。

#### 7.7 其他

1. Fetch adjancent data in the chunk

```
# Fetch adjacent data in the chunk -> Increases speed when loading 
# sequantially or if a tensor's data fits in the cache.
numeric_label = ds.labels[i].numpy(fetch_chunks = True)
```

> Note: If ``True``, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved. This will always be ``True`` even if specified as ``False`` in the following cases: (1) The tensor is ChunkCompressed. (2) The chunk which is being accessed has more than 128 samples.
