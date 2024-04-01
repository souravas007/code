class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        result = []

        def subset_helper(current_nums, index):
            if index == len(nums):
                result.append(current_nums.copy())
                return
            current_nums.append(nums[index])
            subset_helper(current_nums, index + 1)
            current_nums.pop()
            subset_helper(current_nums, index + 1)

        subset_helper([], 0)
        return result
