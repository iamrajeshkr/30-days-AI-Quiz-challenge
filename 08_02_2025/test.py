def maxSubArray(nums):
    maxi=float('-inf')
    sum1=0
    for i in nums :
        sum1+=i
        maxi=max(sum1,maxi)
        if sum1<0:
            sum1=0
    return maxi
print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))