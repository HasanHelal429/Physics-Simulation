
def scmInsertionSort(p1,scm1):
    n = len(scm1)  # Get the length of the array
     
    if n <= 1:
        return  # If the array has 0 or 1 element, it is already sorted, so return

    for i in range(1, n):  # Iterate over the array starting from the second element
        key1 = scm1[i]  # Store the current element as the key to be inserted in the right position
        key2 = p1[i]
        j = i-1
        while j >= 0 and key1 < scm1[j]:  # Move elements greater than key one position ahead
            scm1[j+1] = scm1[j]  # Shift elements to the right
            p1[j+1] = p1[j]
            j -= 1
        scm1[j+1] = key1  # Insert the key in the correct position   
        p1[j+1]=key2
