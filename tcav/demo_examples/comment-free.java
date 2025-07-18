public Map<String, Integer> processStrings(List<String> input) {
    if (input == null) {
        throw new IllegalArgumentException("Input list cannot be null");
    }
    Map<String, Integer> result = new LinkedHashMap<>();
    Set<String> seen = new HashSet<>();
    for (String str : input) {
        if (str == null) {
            continue;
        }
        if (seen.contains(str)) {
            continue;
        }
        seen.add(str);
        String processedStr = str;
        if (isPalindrome(str)) {
            processedStr = new StringBuilder(str).reverse().toString();
        }
        int score = calculateComplexity(processedStr);
        result.put(processedStr, score);
    }
    return result;
}
private boolean isPalindrome(String s) {
    if (s == null) return false;
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s.charAt(left++) != s.charAt(right--)) return false;
    }
    return true;
}
private int calculateComplexity(String s) {
    if (s == null) return 0;
    Set<Character> uniqueChars = new HashSet<>();
    for (char c : s.toCharArray()) {
        uniqueChars.add(c);
    }
    int multiplier = isPalindrome(s) ? 2 : 1;
    return uniqueChars.size() * multiplier;
}