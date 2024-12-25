import jieba


# 全模式使用__cut_all，生成DAG，输出所有可能的切分结果
# 精确模式使用__cut_DAG，生成DAG，根据使用动态规划算法，计算最大概率路径，输出最优切分结果
# __cut_DAG中使用了HMM，__cut_DAG_NO_HMM、__cut_all中没有使用HMM，其中HMM来解决为登录词的问题，使用维特比算法。

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式