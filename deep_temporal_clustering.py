def get_label(idx, totlen):
    # 如果是26键入字母
    result = [0 for _ in range(totlen)]
    if idx in [1, 2, 7, 8]:
        for i in range(26):
            start_index = i * (totlen // 26)  # 起始索引
            end_index = (i + 1) * (totlen // 26)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = i  # 赋值为字母的 ASCII 码
            if i == 25:
                for k in range(end_index, totlen):
                    result[k] = i
        #print(result)
        return result
    if idx == 3:
        for i in range(26 * 5):
            start_index = i * (totlen // 130)  # 起始索引
            end_index = (i + 1) * (totlen // 130)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = i % 26
            if i == 26 * 5:
                for k in range(end_index, totlen):
                    result[k] = 25
        return result
    if idx in [4, 5, 9]:
        str = "privacyiscriticalforensuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhatbeingtypescouldbepasswordsorprivacysensitiveinformation"
        strlen = len(str)
        for i in range(strlen):
            start_index = i * (totlen // strlen)  # 起始索引
            end_index = (i + 1) * (totlen // strlen)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = ord(str[i]) - ord('a')
            if i == strlen:
                for k in range(end_index, totlen):
                    result[k] = ord(str[i]) - ord('a')
        return result
    if idx == 6:
        str = "privacyiscriticalforenuringthesecurityofcomputersystemsandtheprivacyofhumanusersaswhaybeingtypescouldbepasswordsorprivacysensitivesinformationtheresearchcommunityhassutdiedvariouswaystorecognizekeystrokeswhichcanbeclassifiedintothreecategoroesacousticemissionbasedapproacheselectromagneticemmisionbasedapproachesandvisionbasedapprachesacousticemmissionabasedapproachesrecognizekeystrokesbasedontethiertheobservationthattypingsoundsortheobservationthattheacousticemanationfromdifferentkeysarribeaydirrerenttimeasthekeysarelocatedatdifferentplacesinakeyboardelectromagneticemmissionbasedapproachesrecognizekeystrokesbasedontheobsrvationthattheelecyromagneticemanationsfromtheelectrivalvircuitunderneathdifferentkeysinakeyboardaredifferentvisionbasedapproachesrecognizekeystrokeusingvisiontechnologies"
        strlen = len(str)
        for i in range(strlen):
            start_index = i * (totlen // strlen)  # 起始索引
            end_index = (i + 1) * (totlen // strlen)  # 结束索引（不包含）
            for k in range(start_index, end_index):
                result[k] = ord(str[i]) - ord('a')
            #print(ord(str[i]) - ord('a'))
            if i == strlen:
                for k in range(end_index, totlen):
                    result[k] = ord(str[i]) - ord('a')
        #print(result)
        return result