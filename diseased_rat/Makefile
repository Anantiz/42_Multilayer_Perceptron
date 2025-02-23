CXX:=c++
CPPFLAGS:= -Wall -Wextra -Wshadow -I .
DEBUG_FLAGS:= -g3 -fsanitize=address -fsanitize=undefined -fsanitize=leak -fno-omit-frame-pointer`

LIB_NAME = mlp.so
SRC_FILES = mlp.cpp

SRC_OBJECTS := $(patsubst %.cpp, %.o, $(SRC_FILES))

all: $(LIB_NAME) test

$(LIB_NAME): $(SRC_OBJECTS)
	$(CXX) $^ -o $@ -fPIC $(CPPFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CPPFLAGS)

test: test.cpp
	$(CXX) $< -o test_lib_bin $(CPPFLAGS) -L. -lmlp
	./test_lib_bin

clean:
	@$(RM) $(SRC_OBJECTS)

fclean: clean
	@$(RM) $(LIB_NAME)

re: fclean all

.PHONY: all clean fclean re
