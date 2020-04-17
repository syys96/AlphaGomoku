/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED

#include "config.h"

#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class FastBoard {
    friend class FastState;
public:
    /*
        number of vertices in a "letterboxed" board representation
    */
    static constexpr int NUM_VERTICES = ((BOARD_SIZE + 2) * (BOARD_SIZE + 2));

    /*
        no applicable vertex
    */
    static constexpr int NO_VERTEX = 0;
    /*
        vertex of a pass
    */
    static constexpr int PASS   = -1;
    /*
        vertex of a "resign move"
    */
    static constexpr int RESIGN = -2;

    /*
        possible contents of a vertex
    */
    enum vertex_t : char {
        BLACK = 0, WHITE = 1, EMPTY = 2, INVAL = 3
    };
    enum row_t : char {
        LIVE_TREE = 0, CONG_THREE = 1, LIVE_FOUR = 2, CONG_FOUR = 3, FIVE = 4, TOOLONG = 5,
        OTHER_TYPE = 6
    };

    int get_boardsize() const;
    vertex_t get_state(int x, int y) const;
    vertex_t get_state(int vertex) const ;
    int get_vertex(int x, int y) const;
    void set_state(int x, int y, vertex_t content);
    void set_state(int vertex, vertex_t content);
    std::pair<int, int> get_xy(int vertex) const;

    /// 五子棋的禁手
    bool is_forbidden(int vertex, vertex_t color) const;

    /// 五子棋的输赢函数
    float end_score() const;

    bool black_to_move() const;
    bool white_to_move() const;
    int get_to_move() const;
    void set_to_move(int color);

    std::string move_to_text(int move) const;
    int text_to_move(std::string move) const;
    std::string move_to_text_sgf(int move) const;

    void reset_board(int size);
    void display_board(int lastmove = -1);

    bool game_end() const;

protected:

    std::array<vertex_t, NUM_VERTICES>         m_state;      /* board contents */
    /// 8个方向用于五子棋判断棋型
    std::array<int, 8>                         m_dirs;       /* movement directions 8 way */

    int m_tomove;
    int m_numvertices;

    int m_boardsize;
    int m_sidevertices;

    bool has_end = false;
    vertex_t winner = EMPTY;
    int empty_cnt = NUM_INTERSECTIONS;

    bool in_table(int vertex) const;
    void update_continue_info(int vertex, vertex_t content);

    void print_columns();
};

#endif
