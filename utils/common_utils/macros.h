#pragma once

#ifndef PROP_SET
#define PROP_SET(prop_name, type, member_name, size_to_check) \
  \ 
void set_##prop_name(const type& member_name) {               \
    CHECK_EQ(member_name.size(), size_to_check);              \
    this->member_name = member_name;                          \
  }
#endif