/**
 * Processes a list of strings by removing duplicates, 
 * reversing palindromes, and computing a complexity score for each string.
 * 
 * @param input List of input strings
 * @return Map with processed strings as keys and their complexity scores as values
 * @throws IllegalArgumentException if the input list is null
 */
public Map<String, Integer> processStrings(List<String> input) {
    if (input == null) {
        throw new IllegalArgumentException("Input list cannot be null");
    }

    Map<String, Integer> result = new LinkedHashMap<>();
    Set<String> seen = new HashSet<>();

    // Iterate through the input list
    for (String str : input) {
        if (str == null) {
            // Skip null entries
            continue;
        }
        // Block comment:
        /*
         * Check for duplicates. If already processed, skip.
         * Using a Set for O(1) lookup.
         */
        if (seen.contains(str)) {
            continue;
        }
        seen.add(str);

        String processedStr = str;
        // Check if the string is a palindrome
        if (isPalindrome(str)) {
            // If it's a palindrome, reverse it
            processedStr = new StringBuilder(str).reverse().toString();
        }
        // Calculate the complexity score
        int score = calculateComplexity(processedStr);

        // Add to the result map
        result.put(processedStr, score);
    }
    return result;
}

/**
 * Checks if a string is a palindrome.
 * @param s The string to check
 * @return true if palindrome, false otherwise
 */
private boolean isPalindrome(String s) {
    // null strings are not palindromes
    if (s == null) return false;
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s.charAt(left++) != s.charAt(right--)) return false;
    }
    return true;
}

/**
 * Calculates a complexity score for a string.
 * Complexity is defined as:
 *   (number of unique characters) * (2 if palindrome, 1 otherwise)
 * @param s The string
 * @return The complexity score
 */
private int calculateComplexity(String s) {
    if (s == null) return 0;
    Set<Character> uniqueChars = new HashSet<>();
    for (char c : s.toCharArray()) {
        uniqueChars.add(c);
    }
    int multiplier = isPalindrome(s) ? 2 : 1;
    return uniqueChars.size() * multiplier;
}