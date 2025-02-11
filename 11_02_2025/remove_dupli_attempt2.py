def removeDuplicates(nums):
    dupli_set=set()
    i=0
    while i<=len(nums)-1:
        if nums[i] in dupli_set:
            nums.pop(i)
        else:
            dupli_set.add(nums[i])
            i+=1

    return len(nums)

print(removeDuplicates([1,1,2]))