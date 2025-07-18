#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_LEN 1000
#define MAX_STRINGS 100

bool is_palindrome(const char *s) {
    if (!s) return false;
    int left = 0, right = strlen(s) - 1;
    while (left < right) {
        if (s[left++] != s[right--]) return false;
    }
    return true;
}

int calculate_complexity(const char *s) {
    if (!s) return 0;
    int unique[256] = {0};
    int count = 0;
    for (int i = 0; s[i]; i++) {
        if (!unique[(unsigned char)s[i]]) {
            unique[(unsigned char)s[i]] = 1;
            count++;
        }
    }
    int multiplier = is_palindrome(s) ? 2 : 1;
    return count * multiplier;
}

void process_strings(char *input[], int n, char *out_str[], int *out_score, int *out_n) {
    int seen_idx = 0;
    char *seen[MAX_STRINGS];
    int result_idx = 0;

    for (int i = 0; i < n; i++) {
        char *str = input[i];
        if (!str) continue;
        int dup = 0;
        for (int j = 0; j < seen_idx; j++) {
            if (strcmp(seen[j], str) == 0) {
                dup = 1;
                break;
            }
        }
        if (dup) continue;
        seen[seen_idx++] = str;

        char processed[MAX_LEN];
        if (is_palindrome(str)) {
            int len = strlen(str);
            for (int k = 0; k < len; k++) {
                processed[k] = str[len - 1 - k];
            }
            processed[len] = '\0';
        } else {
            strcpy(processed, str);
        }
        out_str[result_idx] = strdup(processed);
        out_score[result_idx] = calculate_complexity(processed);
        result_idx++;
    }
    *out_n = result_idx;
}