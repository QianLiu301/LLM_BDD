//==============================================================================
// Testbench: alu_8bit
// Generated: 2025-12-05 19:51:16
// Generator: testbench_generator.py (deterministic, no LLM)
//
// DUT Module: alu_8bit
// Bitwidth: 8-bit
// Test cases: 20
// Number format: decimal
//==============================================================================

`timescale 1ns / 1ps

module alu_8bit_tb;

    //--------------------------------------------------------------------------
    // Test Signals
    //--------------------------------------------------------------------------
    reg  [7:0] a;
    reg  [7:0] b;
    reg  [3:0] opcode;
    wire [7:0] result;
    wire zero, carry, overflow, negative;

    // Test counters
    integer passed = 0;
    integer failed = 0;
    integer total = 0;

    //--------------------------------------------------------------------------
    // Device Under Test (DUT)
    //--------------------------------------------------------------------------
    alu_8bit uut (
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
        $display("Testbench: alu_8bit");
        $display("DUT: alu_8bit");
        $display("========================================");
        $display("");

        // Test 1: ADD
        a = 8'd10;
        b = 8'd5;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd15);
            failed = failed + 1;
        end

        // Test 2: ADD
        a = 8'd58;
        b = 8'd58;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd116) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd116);
            failed = failed + 1;
        end

        // Test 3: ADD
        a = 8'd92;
        b = 8'd0;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd92) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd92);
            failed = failed + 1;
        end

        // Test 4: ADD
        a = 8'd83;
        b = 8'd107;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd190) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd190);
            failed = failed + 1;
        end

        // Test 5: ADD
        a = 8'd0;
        b = 8'd72;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd72) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd72);
            failed = failed + 1;
        end

        // Test 6: SUB
        a = 8'd10;
        b = 8'd5;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd5) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd5);
            failed = failed + 1;
        end

        // Test 7: SUB
        a = 8'd93;
        b = 8'd93;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd0);
            failed = failed + 1;
        end

        // Test 8: SUB
        a = 8'd34;
        b = 8'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd34) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd34);
            failed = failed + 1;
        end

        // Test 9: SUB
        a = 8'd14;
        b = 8'd105;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd165) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd165);
            failed = failed + 1;
        end

        // Test 10: SUB
        a = 8'd123;
        b = 8'd58;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd65) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd65);
            failed = failed + 1;
        end

        // Test 11: AND
        a = 8'd10;
        b = 8'd5;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd0);
            failed = failed + 1;
        end

        // Test 12: AND
        a = 8'd122;
        b = 8'd122;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd122) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd122);
            failed = failed + 1;
        end

        // Test 13: AND
        a = 8'd46;
        b = 8'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd0);
            failed = failed + 1;
        end

        // Test 14: AND
        a = 8'd61;
        b = 8'd5;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd5) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd5);
            failed = failed + 1;
        end

        // Test 15: AND
        a = 8'd80;
        b = 8'd87;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd80) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd80);
            failed = failed + 1;
        end

        // Test 16: OR
        a = 8'd10;
        b = 8'd5;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd15);
            failed = failed + 1;
        end

        // Test 17: OR
        a = 8'd116;
        b = 8'd116;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd116) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd116);
            failed = failed + 1;
        end

        // Test 18: OR
        a = 8'd27;
        b = 8'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd27) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd27);
            failed = failed + 1;
        end

        // Test 19: OR
        a = 8'd43;
        b = 8'd58;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd59) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd59);
            failed = failed + 1;
        end

        // Test 20: OR
        a = 8'd122;
        b = 8'd8;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 8'd122) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 8'd122);
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
        $dumpfile("alu_8bit_tb.vcd");
        $dumpvars(0, alu_8bit_tb);
    end

endmodule

//==============================================================================
// End of Testbench
//==============================================================================