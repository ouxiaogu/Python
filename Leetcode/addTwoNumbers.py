# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:27:13 2017

@author: peyang
"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class LinkedList(object):
    def __init__(self, lst):
        self.ln = self.__genLinkedList(lst)
    def __genLinkedList(self, lst):
        lns = [ListNode(i) for i in lst]
        for i in range(len(lns)-1):
            lns[i].next = lns[i+1]
        return lns[0]

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        def lnToNum(ln):
            if ln.next is None:
                return ln.val
            return ln.val + 10*lnToNum(ln.next)

        def lenLn(ln):
            len = 1
            while ln.next:
                ln = ln.next
                len += 1
            return len

        def numToLn(num):
            if num/10 == 0:
                return ListNode(num)
            tmp = ListNode(num%10)
            tmp.next = numToLn(num/10)
            return tmp
        len1, len2 = lenLn(l1), lenLn(l2)
        print len1, len2
        num1, num2 = lnToNum(l1), lnToNum(l2)
        print num1, num2
        return numToLn(num1+num2)


if __name__ == "__main__":
    l1 = LinkedList([9, 8]).ln
    l2 = LinkedList([0]).ln
    s = Solution()
    s.addTwoNumbers(l1, l2)