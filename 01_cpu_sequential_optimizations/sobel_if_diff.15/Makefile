# --- Compiler & Flags ---
CC       = icx 
CFLAGS   = -Wall
LDFLAGS  = -lm

# --- Files (Hard-coded for this directory) ---
SRC = sobel_if_diff.c
EXE_O0 = sobel_if_diff_O0
EXE_O3 = sobel_if_diff_O3

# --- Default Target ---
all: $(EXE_O0) $(EXE_O3)

# --- Build Rules ---
$(EXE_O0): $(SRC)
	@echo "Compiling $(SRC) -> $(EXE_O0) (O0) [Μέσα στο sobel_if_diff]"
	@$(CC) $(CFLAGS) -O0 $(SRC) -o $(EXE_O0) $(LDFLAGS)

$(EXE_O3): $(SRC)
	@echo "Compiling $(SRC) -> $(EXE_O3) (O3) [Μέσα στο sobel_if_diff]"
	@$(CC) $(CFLAGS) -O3 $(SRC) -o $(EXE_O3) $(LDFLAGS)

# --- Clean up ---
clean:
	@echo "Cleaning up local binaries in sobel_if_diff..."
	@rm -f $(EXE_O0) $(EXE_O3)