from sklearn import preprocessing
import pandas as pd
import numpy as np

enc = preprocessing.OneHotEncoder()  # 相关onehot的包


# 独热编码
def set_OneHotEncoder(data, colname, start_index, end_index):
    '''
    data -- [[1,2,3,4,7],[0,5,6,8,9]]
    start_index -- 起始列位置索引
    end_index -- 结束列位置索引. 如start_index为1，end_index为3，则取出来的为[[2,3,4],[5,6,8]]
    '''
    if type(data) == pd.core.frame.DataFrame:
        data = np.array(data).tolist()
    if type(data) != list:
        return 'Error dataType, expect list but ' + str(type(data))
    _data, _colname = [line[:start_index] for line in data], colname[:start_index]
    data_, colname_ = [line[end_index + 1:] for line in data], colname[end_index + 1:]

    data = [line[start_index:end_index + 1] for line in data]
    data = pd.DataFrame(data)
    data.columns = colname[start_index:end_index + 1]
    enc.fit(data)
    x_ = enc.transform(data).toarray()  # 已生成
    x_ = [list(line) for line in x_]
    # 加栏目名
    new_columns = []
    for col in data.columns:
        dd = sorted(list(set(list(data[col].astype(str)))))  # 去重并根据升序排列
        for line in dd:
            new_columns.append(str(col) + '#' + str(line))

    end_x = list(map(lambda x, y, z: x + y + z, _data, x_, data_))
    end_columns = list(_colname) + new_columns + list(colname_)
    x__ = pd.DataFrame(end_x, columns=end_columns)
    return x__  # 返回数据框形式


# 哑变量
# 对性别、职业等因子变量，构造其哑变量(虚拟变量)
def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')  # 转换成数据类别类型，pandas用法
        dummy = pd.get_dummies(data[col])  # get_dummies为pandas里面求哑变量的包
        dummy = dummy.add_prefix('{}#'.format(col))  # add_prefix为加上前缀
        data.drop(col, axis=1, inplace=True)
        data = data.join(dummy)  # index即为userid，所以可以用join
    return data


if __name__ == '__main__':
    # xlst = [[0, 2, 1, 3, 4, 5], [9, 1, 1, 4, 5, 6], [8, 9, 2, 3, 4, 6], [8, 11, 23, 56, 78, 99]]
    # x = pd.DataFrame(xlst)
    # x.columns = ['a', 'b', 'c', 'd', 'e', 'f']
    # # y = [1, 0, 1, 1, 1]
    # print('----------------------------以下为onehot----------------------------------')
    # print(set_OneHotEncoder(x, x.columns, 2, 4))
    # df = pd.read_csv('home.csv')
    # df_onehot_12_16 = set_OneHotEncoder(df, df.columns, 11, 15)
    # print(df_onehot_12_16.columns)
    # df_onehot_12_16.to_csv('home_onehot_12_16.csv')

    pd.set_option('display.max_columns',  1000000)
    df = pd.read_csv('home_onehot_12_16.csv')
    print( set(df['OCCUPATION_TYPE']))
    # cnt = 0
    # for i in df.columns:
    #     if cnt % 7 == 0:
    #         print('')
    #     print('\'' + i + '\'', end=', ')
    #     cnt += 1
