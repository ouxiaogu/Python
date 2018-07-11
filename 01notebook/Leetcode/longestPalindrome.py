class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """

        '''
        # solution 1
        Loop every location i
        use i as start point:
            left_start=i, right_start=j=i+k, s[i]=s[i+x]|x=1,2,..k,
            compare s[i-1], s[j+1] until unequal, get current longestPalindrome
        '''
        if len(s) <= 1:
            return s
        longest_pldr = ''
        longest_len = 0
        for i in range(len(s)):
            left_start = i
            k = 1
            while i+k<len(s) and s[i+k] == s[i]:
                k += 1
            right_start= i+k-1
            steps = 1
            while left_start-steps>=0 and right_start+steps<len(s) and s[left_start-steps] == s[right_start+steps]:
                steps += 1
            current_pldr = s[left_start-steps+1:right_start+steps]
            if len(current_pldr) > longest_len:
                longest_len = len(current_pldr)
                longest_pldr = current_pldr
        return longest_pldr
        '''
        # solution 2
        Loop head pos i: 0->n-1
            Loop tail pos j n-1->0
                if is Palindrome:
                    if longer than > current longestPalindrome:
                        swap
        '''

        '''
        '''

if __name__ == '__main__':
    s = Solution()
    print s.longestPalindrome('babadfasafdasdfasdffaf')
    print s.longestPalindrome('bb')