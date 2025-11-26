//==============================================================================
// ALU Module: alu_32bit
// Bitwidth: 32-bit
// Generated: 2025-11-27 00:50:07
// Generator: alu_generator.py (deterministic, no LLM)
//
// This is a synthesizable 32-bit ALU module generated from specification.
// All operations are purely combinational.
//==============================================================================

`timescale 1ns / 1ps

module alu_32bit (
    // Inputs
    input  wire [31:0] a,           // First operand
    input  wire [31:0] b,           // Second operand
    input  wire [3:0]  opcode,      // Operation select

    // Outputs
    output reg  [31:0] result,      // Operation result
    output reg         zero,        // Zero flag (result == 0)
    output reg         carry,       // Carry/borrow flag
    output reg         overflow,    // Overflow flag (signed)
    output reg         negative     // Negative flag (MSB of result)
);

    //--------------------------------------------------------------------------
    // Internal Signals
    //--------------------------------------------------------------------------
    reg [32:0] result_full;  // Extended result for carry detection

    // Operation opcodes (from specification)
    localparam OP_ADD = 4'b0000;
    localparam OP_SUB = 4'b0001;
    localparam OP_AND = 4'b0010;
    localparam OP_OR  = 4'b0011;
    localparam OP_XOR = 4'b0100;
    localparam OP_NOT = 4'b0101;
    localparam OP_SHL = 4'b0110;
    localparam OP_SHR = 4'b0111;

    //--------------------------------------------------------------------------
    // ALU Operation Logic (Combinational)
    //--------------------------------------------------------------------------
    always @(*) begin
        // Default values
        result_full = {33'b0};

        case (opcode)
            // Addition (A + B)
            4'b0000: begin
                result_full = a + b;
            end

            // Subtraction (A - B)
            4'b0001: begin
                result_full = a - b;
            end

            // Bitwise AND (A & B)
            4'b0010: begin
                result_full = {1'b0, a & b};
            end

            // Bitwise OR (A | B)
            4'b0011: begin
                result_full = {1'b0, a | b};
            end

            // Default: No operation
            default: begin
                result_full = {33'b0};
            end
        endcase
    end

    //--------------------------------------------------------------------------
    // Flag Logic
    //--------------------------------------------------------------------------
    always @(*) begin
        // Extract result (lower 32 bits)
        result = result_full[31:0];

        // Zero flag: set when result is zero
        zero = (result == 32'b0);

        // Carry flag: set from extended bit (for ADD/SUB)
        carry = result_full[32];

        // Negative flag: set when MSB is 1 (signed interpretation)
        negative = result[31];

        // Overflow flag: signed overflow detection
        // For ADD: overflow when both operands have same sign but result has different sign
        // For SUB: overflow when operands have different signs and result sign != a's sign
        case (opcode)
            OP_ADD: overflow = (a[31] == b[31]) && (result[31] != a[31]);
            OP_SUB: overflow = (a[31] != b[31]) && (result[31] != a[31]);
            default: overflow = 1'b0;
        endcase
    end

endmodule

//==============================================================================
// End of ALU Module
//==============================================================================
