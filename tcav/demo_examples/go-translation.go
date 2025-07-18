package main

import (
    "fmt"
)

func isPalindrome(s string) bool {
    n := len(s)
    for i := 0; i < n/2; i++ {
        if s[i] != s[n-1-i] {
            return false
        }
    }
    return true
}

func calculateComplexity(s string) int {
    unique := make(map[rune]bool)
    for _, c := range s {
        unique[c] = true
    }
    multiplier := 1
    if isPalindrome(s) {
        multiplier = 2
    }
    return len(unique) * multiplier
}

func processStrings(input []string) map[string]int {
    result := make(map[string]int)
    seen := make(map[string]bool)
    for _, s := range input {
        if s == "" {
            continue
        }
        if seen[s] {
            continue
        }
        seen[s] = true
        processed := s
        if isPalindrome(s) {
            // Reverse string
            runes := []rune(s)
            for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
                runes[i], runes[j] = runes[j], runes[i]
            }
            processed = string(runes)
        }
        score := calculateComplexity(processed)
        result[processed] = score
    }
    return result
}