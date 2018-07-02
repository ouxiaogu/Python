class Solution(object):
    def lengthOfLongestSubstring1(self, s):
        """
        :type s: str
        :rtype: int
        """
        def isUniqueStr(s):
            a = {}
            for char in list(s):
                if a.has_key(char):
                    return False
                a[char] = 1
            return True

        lenght = len(s)
        longestStr = ''
        longestLen = 0
        for i in range(lenght):
            if lenght-i <= longestLen:
                continue
            for j in range(longestLen+i+1, lenght+1):
                curSubStr = s[i:j]
                if isUniqueStr(curSubStr):
                    if longestLen < len(curSubStr):
                        longestLen = len(curSubStr)
                        longestStr = curSubStr
        return longestLen, longestStr
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        curL = ""
        head = 0
        longest = 0
        for i in range(len(s)):
            if s[i] in curL:
                rel_idx = curL.index(s[i])
                head += rel_idx+1
                curL = s[head:(i+1)]
            else:
                curL = s[head:(i+1)]
            longest = max(longest, len(curL))
        return longest

    def run(self):
        #s = 'mgvqizxqvtrgajgzdmbgfvzctobhozvdfqtnrsgnlxvnidmlppsukryghbnxaiaf'
        s = 'nbumdladpycygtrgutpdzlajzovccwcqaycfjeibclzkgsqanifmtfxsusuyqzoqxsy'
        s = ''
        print self.lengthOfLongestSubstring(s)


s = Solution()
s.run()