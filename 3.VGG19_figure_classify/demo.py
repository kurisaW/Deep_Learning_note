# 测试 生成提交submission.csv
def submission(net, test_loader):
    net.load_state_dict(torch.load('model.params'))
    net = net.to(device)
    result = []
    with torch.no_grad():  # 不更新梯度
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            pre = net(images)
            # 获取 图片是狗的概率
            predicted = torch.nn.functional.softmax(pre, dim=1)[:, 1]
            # 把预测结果用 list保存
            for j in range(len(predicted)):
                result.append({
                    "id": labels[j].item(),
                    "label": predicted[j].item()
                })
    # 把 list 转成 dataframe 然后保存为csv文件
    columns = result[0].keys()
    print(columns)
    result_dict = {col: [anno[col] for anno in result] for col in columns}
    print(result_dict)
    result_file = pandas.DataFrame(result_dict)
    result_file = result_file.sort_values("id")
    result_file.to_csv('./submission.csv', index=None)
submission(net, test_loader)
