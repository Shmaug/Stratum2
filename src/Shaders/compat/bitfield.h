#ifndef BITFIELD_H
#define BITFIELD_H

// https://gist.github.com/Jeff-Russ/c9b471158fa7427280e6707d9b11d7d2

/* Bit Manipulation Macros
A good article: http://www.coranac.com/documents/working-with-bits-and-bitfields/
x    is a variable that will be modified.
y    will not.
pos  is a unsigned int (usually 0 through 7) representing a single bit position where the
     right-most bit is bit 0. So 00000010 is pos 1 since the second bit is high.
bm   (bit mask) is used to specify multiple bits by having each set ON.
bf   (bit field) is similar (it is in fact used as a bit mask) but it is used to specify a
     range of neighboring bit by having them set ON.
*/
/* shifts left the '1' over pos times to create a single HIGH bit at location pos. */
#define BIT(pos) ( 1u << (pos) )

/* Set single bit at pos to '1' by generating a mask
in the proper bit location and ORing x with the mask. */
#define SET_BIT(x, pos) ( (x) |= (BIT(pos)) )
#define SET_BITS(x, bm) ( (x) |= (bm) ) // same but for multiple bits

/* Set single bit at pos to '0' by generating a mask
in the proper bit location and ORing x with the mask. */
#define UNSET_BIT(x, pos) ( (x) &= ~(BIT(pos)) )
#define UNSET_BITS(x, bm) ( (x) &= (~(bm)) ) // same but for multiple bits

/* Set single bit at pos to opposite of what is currently is by generating a mask
in the proper bit location and ORing x with the mask. */
#define FLIP_BIT(x, pos) ( (x) ^= (BIT(pos)) )
#define FLIP_BITS(x, bm) ( (x) ^= (bm) ) // same but for multiple bits

/* Return '1' if the bit value at position pos within y is '1' and '0' if it's 0 by
ANDing x with a bit mask where the bit in pos's position is '1' and '0' elsewhere and
comparing it to all 0's.  Returns '1' in least significant bit position if the value
of the bit is '1', '0' if it was '0'. */
#define CHECK_BIT(y, pos) ( ( 0u == ( (y)&(BIT(pos)) ) ) ? 0u : 1u )
#define CHECK_BITS_ANY(y, bm) ( ( (y) & (bm) ) ? 0u : 1u )
// warning: evaluates bm twice:
#define CHECK_BITS_ALL(y, bm) ( ( (bm) == ((y)&(bm)) ) ? 0u : 1u )

// These are three preparatory macros used by the following two:
#define SET_LSBITS(len) ( (1u << (len)) - 1u ) // the first len bits are '1' and the rest are '0'
#define BF_MASK(start, len) ( SET_LSBITS(len) << (start) ) // same but with offset
#define BF_PREP(y, start, len) ( ((y)&SET_LSBITS(len)) << (start) ) // Prepare a bitmask

/* Extract a bitfield of length len starting at bit start from y. */
#define BF_GET(y, start, len) ( ((y) >> (start)) & SET_LSBITS(len) )

/* Insert a new bitfield value bf into x. */
#define BF_SET(x, bf, start, len) ( x = ((x) &~ BF_MASK(start, len)) | BF_PREP(bf, start, len) )

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif