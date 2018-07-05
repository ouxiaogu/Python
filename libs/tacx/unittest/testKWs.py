def info(list_, spacing=10, collapse=1):
    sp = [' ' for ii in range(spacing)]
    sp = ''.join(sp)
    endl = ['\n' for ii in range(collapse)]
    endl = ''.join(endl)
    list_ = map(str, list_)
    print (sp+endl).join(list_)

if __name__ == '__main__':
    m_list = [1, 2]
    info(m_list)
    info(m_list, 12)
    info(m_list, collapse=0)
    # info(spacing=15, m_list)
