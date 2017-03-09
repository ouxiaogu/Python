class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        # merge sorted lists
        i = j = 0
        merged = []
        while True:
            if i == len(nums1):
                merged.extend(nums2[j:])
                break
            elif j == len(nums2):
                merged.extend(nums1[i:])
                break
            elif nums1[i] <= nums2[j]:
                merged.append(nums1[i])
                i += 1
            else:
                merged.append(nums2[j])
                j += 1

        if len(merged)%2 == 0:
            return 0.5*(merged[len(merged)/2-1]+merged[len(merged)/2])
        else:
            return 1.0*merged[len(merged)/2]

    def run(self):
        nums1 = [1, 2]
        nums2 = [3, 4]
        print self.findMedianSortedArrays(nums1, nums2)


s = Solution()
s.run()
