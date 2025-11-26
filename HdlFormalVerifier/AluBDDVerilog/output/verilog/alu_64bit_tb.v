//==============================================================================
// Testbench: alu_64bit
// Generated: 2025-11-27 00:50:58
// Generator: testbench_generator.py (deterministic, no LLM)
//
// DUT Module: alu_64bit
// Bitwidth: 64-bit
// Test cases: 20
// Number format: decimal
//==============================================================================

`timescale 1ns / 1ps

module alu_64bit_tb;

    //--------------------------------------------------------------------------
    // Test Signals
    //--------------------------------------------------------------------------
    reg  [63:0] a;
    reg  [63:0] b;
    reg  [3:0] opcode;
    wire [63:0] result;
    wire zero, carry, overflow, negative;

    // Test counters
    integer passed = 0;
    integer failed = 0;
    integer total = 0;

    //--------------------------------------------------------------------------
    // Device Under Test (DUT)
    //--------------------------------------------------------------------------
    alu_64bit uut (
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result),
        .zero(zero),
        .carry(carry),
        .overflow(overflow),
        .negative(negative)
    );

    //--------------------------------------------------------------------------
    // Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("Testbench: alu_64bit");
        $display("DUT: alu_64bit");
        $display("========================================");
        $display("");

        // Test 1: ADD
        a = 64'd10;
        b = 64'd5;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd15);
            failed = failed + 1;
        end

        // Test 2: ADD
        a = 64'd118;
        b = 64'd118;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd236) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd236);
            failed = failed + 1;
        end

        // Test 3: ADD
        a = 64'd92;
        b = 64'd0;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd92) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd92);
            failed = failed + 1;
        end

        // Test 4: ADD
        a = 64'd7753987954478579207;
        b = 64'd6017043200574059716;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd13771031155052638923) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd13771031155052638923);
            failed = failed + 1;
        end

        // Test 5: ADD
        a = 64'd4993603137919118814;
        b = 64'd8894918318955573194;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd13888521456874692008) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd13888521456874692008);
            failed = failed + 1;
        end

        // Test 6: SUB
        a = 64'd10;
        b = 64'd5;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd5) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd5);
            failed = failed + 1;
        end

        // Test 7: SUB
        a = 64'd400;
        b = 64'd400;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd0);
            failed = failed + 1;
        end

        // Test 8: SUB
        a = 64'd34;
        b = 64'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd34) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd34);
            failed = failed + 1;
        end

        // Test 9: SUB
        a = 64'd1062015355818845848;
        b = 64'd7586402456324509308;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd11922356973203888156) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd11922356973203888156);
            failed = failed + 1;
        end

        // Test 10: SUB
        a = 64'd4228065651077726551;
        b = 64'd4817880938540546958;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd17856928786246731209) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd17856928786246731209);
            failed = failed + 1;
        end

        // Test 11: AND
        a = 64'd10;
        b = 64'd5;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd0);
            failed = failed + 1;
        end

        // Test 12: AND
        a = 64'd628;
        b = 64'd628;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd628) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd628);
            failed = failed + 1;
        end

        // Test 13: AND
        a = 64'd46;
        b = 64'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd0);
            failed = failed + 1;
        end

        // Test 14: AND
        a = 64'd4465834584413815543;
        b = 64'd5785536705484297034;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd1173276626018894402) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd1173276626018894402);
            failed = failed + 1;
        end

        // Test 15: AND
        a = 64'd7676810840197284270;
        b = 64'd7848942153956742598;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd7532554364863529350) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd7532554364863529350);
            failed = failed + 1;
        end

        // Test 16: OR
        a = 64'd10;
        b = 64'd5;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd15);
            failed = failed + 1;
        end

        // Test 17: OR
        a = 64'd580;
        b = 64'd580;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd580) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd580);
            failed = failed + 1;
        end

        // Test 18: OR
        a = 64'd27;
        b = 64'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd27) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd27);
            failed = failed + 1;
        end

        // Test 19: OR
        a = 64'd3099515833590915264;
        b = 64'd631655993265934062;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd3154692374118985454) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd3154692374118985454);
            failed = failed + 1;
        end

        // Test 20: OR
        a = 64'd8873789065998484944;
        b = 64'd8495185410343105479;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 64'd9216333237999499223) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 64'd9216333237999499223);
            failed = failed + 1;
        end

        $display("");
        $display("========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        if (failed == 0)
            $display("[SUCCESS] All tests passed!");
        else
            $display("[FAILURE] Some tests failed!");
        $display("========================================");

        #10;
        $finish;
    end

    //--------------------------------------------------------------------------
    // VCD Dump for Waveform Viewing
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("alu_64bit_tb.vcd");
        $dumpvars(0, alu_64bit_tb);
    end

endmodule

//==============================================================================
// End of Testbench
//==============================================================================