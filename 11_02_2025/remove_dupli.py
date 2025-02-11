def removeDuplicates(nums):
    dupli_set=set()
    for i in range(len(nums)-1):
        for j in range(i+1,len(nums)):
            if nums[i]==nums[j]:
                dupli_set.add(nums[i])
    for i in range(len(nums)):
        for j in dupli_set:
            if nums[i]==j:
                nums.pop(i)
    return nums

print(removeDuplicates([1,1,2]))