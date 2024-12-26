# Nama executable
TARGET = neural_network

# Compiler dan flags
CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm

# File sumber dan objek
SRC = main_nn.c algorithm.c
OBJ = $(SRC:.c=.o)

# Aturan default
all: $(TARGET)

# Membuat executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Membuat file objek
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Membersihkan file yang tidak perlu
clean:
	rm -f $(OBJ) $(TARGET)

# Phony targets
.PHONY: all clean
