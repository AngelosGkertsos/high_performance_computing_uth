# --- Compiler & Flags ---
CC       = icx 
CFLAGS   = -Wall
LDFLAGS  = -lm

# --- Files (Hard-coded for this directory) ---
SRC = sobel_16pixels_func_inling.c
EXE_O0 = sobel_16pixels_func_inling_O0
EXE_O3 = sobel_16pixels_func_inling_O3

# --- Default Target ---
all: $(EXE_O0) $(EXE_O3)

# --- Build Rules ---
$(EXE_O0): $(SRC)
	@echo "Compiling $(SRC) -> $(EXE_O0) (O0) [Μέσα στο sobel_16pixels_func_inling]"
	@$(CC) $(CFLAGS) -O0 $(SRC) -o $(EXE_O0) $(LDFLAGS)

$(EXE_O3): $(SRC)
	@echo "Compiling $(SRC) -> $(EXE_O3) (O3) [Μέσα στο sobel_16pixels_func_inling]"
	@$(CC) $(CFLAGS) -O3 $(SRC) -o $(EXE_O3) $(LDFLAGS)

# --- Clean up ---
clean:
	@echo "Cleaning up local binaries in sobel_16pixels_func_inling..."
	@rm -f $(EXE_O0) $(EXE_O3)