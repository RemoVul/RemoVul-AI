#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//  The vulnerability arises from the fact that memset is called on the password and localToken arrays
//  after they have been freed or are about to go out of scope.
//  Once memory is freed or goes out of scope, it should not be accessed or modified anymore.
//  This is because the memory may be reused for other purposes,
//  and attempting to modify it can lead to unexpected behavior,
//  including security vulnerabilities.
void f(char *password, size_t bufferSize) {
  char localToken[256];
  init(localToken, password);
  memset(password, ' ', strlen(password)); // Noncompliant, password is about to be freed
  memset(localToken, ' ', strlen(localToken)); // Noncompliant, localToken is about to go out of scope
  free(password);
}


//  By using memset_s in the modified code, the compiler or runtime environment can provide
//  additional protections to detect potential buffer overflows.
//  If the specified buffer size is smaller than the length parameter
//  (bufferSize or sizeof(localToken)), it will prevent the overwrite
//  operation from exceeding the bounds of the buffer.

void f(char *password, size_t bufferSize) {
  char localToken[256];
  init(localToken, password);
  memset_s(password, bufferSize, ' ', strlen(password));
  memset_s(localToken, sizeof(localToken), ' ', strlen(localToken));
  free(password);
}



// function return array of char
char *getstr()
{
char src[20] = "C programming";
char dest[20];
// copying src to dest
strcpy(dest, src);
puts(dest); // C programming

    
    
return dest;
}

char *getstr()
{
    char src[20] = "C programming";
    char dest[20];
    // copying src to dest using strncpy  
    // This ensures that the dest buffer is not overflowed
    strncpy(dest, src, sizeof(dest) - 1);
    dest[sizeof(dest) - 1] = '\0'; // Ensure null-termination

    puts(dest); // C programming

    // used to return a dynamically allocated copy of the dest string
    return strdup(dest);
}

int predict(){ 
 char hiden[]="password";
 char guess[20];
 printf("Enter the password : ");
 gets(guess);
 //printf("Your Guess  %s ", guess);
 sprintf(str, "%s", guess);   // Sensitive: `str` buffer size is not checked and it is vulnerable to overflows
 return strcmp(guess,hiden);
} 

int predict(){ 
 char hidden[] = "password";
 char guess[20];
 printf("Enter the password: ");

 if (fgets(guess, sizeof(guess), stdin) != NULL) {
    guess[strcspn(guess, "\n")] = '\0';  // Remove trailing newline if present
 }
 snprintf(str, sizeof(str), "%s", message); // Prevent overflows by enforcing a maximum size for `str` buffer
 return strcmp(guess,hiden);
} 



char *concatstr(char *s1, char *s2){
// strcat concatenates str1 and str2
// the resultant string is stored in s1.
/* strcat() function is used to append a source string to a target string.
However, just like the above-mentioned functions, it doesn’t check the length/boundary.
This can easily lead to a segmentation fault or a buffer overflow attack. */
strcat(s1, s2);
return s1;
}

char *concatstr(char *s1, char *s2) {
    size_t s1_len = strlen(s1);
    size_t s2_len = strlen(s2);
    size_t total_len = s1_len + s2_len;
    
    char *result = malloc(total_len + 1);  // Allocate memory for the concatenated string
    if (result != NULL) {
        strncpy(result, s1, s1_len);  // Copy s1 to the result buffer
        strncpy(result + s1_len, s2, s2_len);  // Concatenate s2 to the result buffer
        
        result[total_len] = '\0';  // Null-terminate the result string
    }
    
    return result;
}

