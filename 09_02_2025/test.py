def sortColors(nums):
    for i in range(len(nums)-1):
        for j in range(i,len(nums)):
            if nums[i]>nums[j]:
                nums[i], nums[j]=nums[j], nums[i]
    return nums

print(sortColors([2,0,2,1,1,0]))