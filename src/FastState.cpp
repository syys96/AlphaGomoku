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

#include "config.h"
#include "FastState.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "FastBoard.h"
#include "Utils.h"
#include "Zobrist.h"
#include "GTP.h"

using namespace Utils;

void FastState::init_game(int size) {
    board.reset_board(size);
    m_movenum = 0;
    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;
    return;
}

void FastState::reset_game() {
    reset_board();
    m_movenum = 0;
    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;
}

void FastState::reset_board() {
    board.reset_board(board.get_boardsize());
}

bool FastState::is_move_legal(int color, int vertex) const {
    return !cfg_analyze_tags.is_to_avoid(color, vertex, m_movenum) && (
              vertex == FastBoard::PASS ||
                 vertex == FastBoard::RESIGN ||
                 (vertex != m_komove &&
                      board.get_state(vertex) == FastBoard::EMPTY &&
                      !board.is_forbidden(vertex, FastBoard::vertex_t(color))));
}

void FastState::play_move(int vertex) {
    play_move(board.m_tomove, vertex);
}

/// 这一层类的落子函数
void FastState::play_move(int color, int vertex) {
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];
    if (vertex == FastBoard::PASS) {
        // No Ko move
        m_komove = FastBoard::NO_VERTEX;
    } else {
        // 落子并返回
        m_komove = board.update_board(color, vertex);
    }
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];

    m_lastmove = vertex;
    m_movenum++;

    if (board.m_tomove == color) {
        board.m_hash ^= Zobrist::zobrist_blacktomove;
    }
    /// 该另一方落子
    board.m_tomove = !color;
}

size_t FastState::get_movenum() const {
    return m_movenum;
}

int FastState::get_last_move() const {
    return m_lastmove;
}

int FastState::get_to_move() const {
    return board.m_tomove;
}

void FastState::set_to_move(int tom) {
    board.set_to_move(tom);
}

void FastState::display_state() {
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    board.display_board(get_last_move());
}

std::string FastState::move_to_text(int move) {
    return board.move_to_text(move);
}

/// 此处可实现五子棋的输赢函数 TODO 3
float FastState::final_score() const {
    return board.end_score();
    /// return board.area_score(get_komi() + get_handicap());
}

std::uint64_t FastState::get_symmetry_hash(int symmetry) const {
    return board.calc_symmetry_hash(m_komove, symmetry);
}
